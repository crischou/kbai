class MonsterClassificationAgent:
    #Non-numeric attributes whose values still have a natural ordering, so the
    #agent can reason about how far a value sits from the ones it has accepted.
    #Numeric attributes (leg-count, horn-count, and so on) are ordered already.
    ORDINAL_SCALES = {
        "size": ["tiny", "small", "medium", "large", "huge"],
    }

    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    def solve(self, samples, new_monster):
        #This agent treats a species as a conjunction of allowed value sets:
        #one set of permitted values per attribute, and a monster belongs to
        #the species when every one of its attributes holds a permitted value.
        #The problem statement licenses exactly this model when it says the
        #attributes are independent (no species is "one horn when yellow but
        #two horns when blue") and that the attributes are all there is.
        #
        #Learning that concept is incremental concept learning over a version
        #space, so the agent works in three stages:
        #
        # 1. Generalize from the positives. Every value seen in a positive is
        #    definitely permitted, which gives the most specific hypothesis
        #    still consistent with the data.
        # 2. Specialize using the negatives. A negative must break at least one
        #    attribute, and it cannot be breaking one whose value a positive
        #    already vouched for. When exactly one attribute is left as a
        #    possible culprit, that value is *proven* forbidden. Proving one
        #    value forbidden can explain away another negative, so this runs to
        #    a fixed point.
        # 3. Guess about what is still unknown. Steps 1 and 2 settle values the
        #    evidence decides outright; the samples are not exhaustive, so
        #    anything left over gets a likelihood instead of a verdict, and the
        #    likelihoods combine across attributes.

        #The problem guarantees that identical parameters mean identical
        #species, so a sample that matches exactly settles the question.
        for monster, label in samples:
            if monster == new_monster:
                return label

        positives = [monster for monster, label in samples if label]
        negatives = [monster for monster, label in samples if not label]

        #With nothing positive to generalize from there is no species to match
        #against, and the exact-match check above has already ruled out the one
        #case the negatives could settle on their own.
        if not positives:
            return False

        #Stage 1: the most specific hypothesis. permitted[attribute] holds the
        #values some positive example has vouched for.
        attributes = set()
        for monster in positives:
            attributes.update(monster.keys())

        permitted = {attribute: set() for attribute in attributes}
        for monster in positives:
            for attribute in attributes:
                permitted[attribute].add(monster.get(attribute))

        #Stage 2: values the negatives prove are forbidden.
        forbidden = self._deduce_forbidden(negatives, attributes, permitted)

        #Stage 3: score the new monster one attribute at a time. An attribute
        #whose value a positive has vouched for contributes nothing against it;
        #a proven-forbidden value rules the monster out on its own; anything
        #else contributes the likelihood that the species tolerates it.
        likelihood = 1.0
        for attribute in attributes:
            value = new_monster.get(attribute)
            if value in permitted[attribute]:
                continue
            if value in forbidden[attribute]:
                return False
            likelihood *= self._tolerance(
                attribute, value, permitted, forbidden,
                len(positives), negatives, attributes
            )

        #Ties break positive: an even split means the evidence never spoke
        #against the monster, and an attribute the samples say nothing about is
        #a poor reason to reject an otherwise matching creature.
        return likelihood >= 0.5

    def _deduce_forbidden(self, negatives, attributes, permitted):
        #A negative example fails on at least one attribute, and it can only be
        #failing on an attribute whose value no positive has vouched for. Narrow
        #each negative to those candidate attributes: if a single candidate is
        #left, its value has to be the reason the example is negative, so that
        #value is forbidden. Newly forbidden values can account for other
        #negatives outright, which is why this repeats until nothing changes.
        forbidden = {attribute: set() for attribute in attributes}

        changed = True
        while changed:
            changed = False
            for monster in negatives:
                candidates = self._unexplained(monster, attributes, permitted)
                #Already accounted for by something known to be forbidden, so
                #this example has nothing further to tell us.
                if any(monster.get(a) in forbidden[a] for a in candidates):
                    continue
                if len(candidates) == 1:
                    attribute = candidates[0]
                    forbidden[attribute].add(monster.get(attribute))
                    changed = True

        return forbidden

    def _unexplained(self, monster, attributes, permitted):
        #Attributes where this monster's value has not been vouched for by any
        #positive example, i.e. the attributes that could be disqualifying it.
        return [a for a in attributes if monster.get(a) not in permitted[a]]

    def _tolerance(self, attribute, value, permitted, forbidden,
                   positive_count, negatives, attributes):
        #Likelihood that the species tolerates an unseen value for one
        #attribute. Three pieces of evidence bear on it.

        #First, how much variety the species has already shown here. An
        #attribute that has taken several different values across the positives
        #is evidently a loose one, so another unseen value is plausible; an
        #attribute pinned to a single value across every positive looks like a
        #defining trait of the species, so departing from it is doubtful. How
        #often the positives merely repeated a value we had already recorded
        #matters too: every repetition is another chance the species had to show
        #us something new and did not, which makes the recorded values look
        #closer to the whole story.
        variety = len(permitted[attribute])
        if variety == 0:
            return 0.5
        repetitions = max(positive_count - variety, 0)
        tolerance = (1.0 - 1.0 / (2.0 * variety)) * (0.98 ** repetitions)

        #Second, where the value falls on an ordered attribute, which says more
        #than bare set membership. A value bracketed by two the species accepts
        #is very likely accepted too: a species with one-horned and three-horned
        #members almost certainly allows two. A value outside that bracket grows
        #less likely the further out it sits, so nine horns is a far worse bet
        #than four when the positives top out at three.
        position = self._ordinal(attribute, value)
        if position is not None:
            known = [self._ordinal(attribute, v) for v in permitted[attribute]]
            known = [p for p in known if p is not None]
            if known:
                low, high = min(known), max(known)
                if low < position < high:
                    tolerance = max(tolerance, 0.9)
                elif position < low or position > high:
                    distance = low - position if position < low else position - high
                    #Measured against the range the species is known to cover,
                    #so a wide-ranging attribute tolerates a wider reach.
                    reach = max(high - low, 1.0)
                    tolerance *= 1.0 / (1.0 + distance / reach)

        #Third, whether negatives argue against this specific value. A negative
        #carrying this value is suspicious, but only weakly so when that example
        #departs from the positives in several other ways too: any one of those
        #departures could be the real reason it is negative. The more ways it
        #differs, the less of the blame this value deserves.
        for monster in negatives:
            if monster.get(attribute) != value:
                continue
            candidates = self._unexplained(monster, attributes, permitted)
            #This example is already explained by a value known to be
            #forbidden, so it is no evidence against the value in hand.
            if any(monster.get(a) in forbidden[a] for a in candidates):
                continue
            if candidates:
                tolerance *= 1.0 - 1.0 / len(candidates)

        return tolerance

    def _ordinal(self, attribute, value):
        #Position of a value on an ordered scale, or None if the attribute is
        #not ordered. Booleans are excluded: Python counts them as ints, but
        #there is nothing "between" True and False.
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        scale = self.ORDINAL_SCALES.get(attribute)
        if scale is not None and value in scale:
            return float(scale.index(value))
        return None
