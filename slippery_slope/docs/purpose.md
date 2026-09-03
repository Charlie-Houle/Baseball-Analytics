### Purpose:
Investigate how an at-bat's trajectory changes based on its early pitch results, in particular whether a first-pitch ball starts a "slippery slope" toward a walk or a more passive approach from the pitcher.

### Core Questions:
- Given the result of pitch 1 (or any count state), what does the rest of the at-bat look like? (walk rate, strikeout rate, contact quality, conditioned on count)
- Does falling behind 1-0 measurably change pitcher behavior for the rest of the at-bat, or just the raw odds of a walk?
- If pitcher behavior does change, does it show up in pitch type selection, pitch location, or both?

### Basis of "Slippery Slope":
- Start with count-state outcome rates: given a count, what's the distribution of eventual at-bat results?
- First-pitch-ball effect: compare at-bats that open 1-0 against at-bats that open 0-1, and see how the downstream numbers diverge
- Pitch-type strategy: extend Baseball Savant's "Pitcher Plinko" framing (how a pitcher's arsenal mix shifts as the count evolves) to isolate the effect of a first-pitch ball specifically, not just count in general
- Location strategy: once type-level patterns are established, check whether pitch location gets more conservative (more center-zone, less edge) after falling behind, per pitcher and leaguewide
- Basic to complex, in that order: results and rates first, arsenal strategy second, location strategy last

### Planned Workflow
1. Baseline count-state outcome rates: walk rate, strikeout rate, and other AB results by count (0-0, 1-0, 0-1, etc.)
2. First-pitch-ball effect: isolate at-bats starting 1-0 vs. 0-1, compare downstream results (walk rate, pitches per PA, etc.)
3. Pitch-type strategy: Pitcher Plinko-style breakdown of arsenal usage by count, before vs. after a first-pitch ball
4. Location strategy: once type-level patterns hold up, check whether pitch location shifts with count and after a first-pitch ball, per pitcher and leaguewide 

### Inspiration
Throwaway code cell from Pitching+ showed that Graham Ashcraft had very good Stuff+ and average Location+, but Ashcraft's recent seasons have had above-average walk rates and limited success compared to other pitchers on the list(e.g. Skenes, Clase). 

ALso inspired by Dustin May, who appears on Pitching Ninja every other week but has found limited MLB success (visibly great Stuff+, but can't translate it into success).