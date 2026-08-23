### Purpose: 
Create "+" style model that scores on following fronts: 
1. Stuff+ (how "filthy" a pitch is)
2. Location+ (given context of AB, was pitch thrown in good location) 
3. Pitching+ (given Stuff+ and Location+, how good was the pitch)
4. bestPitch+ (difference between "optimal" pitch and Pitching+)

### Basis of "Good":

#### Stuff+: 
- "Outlierness" to other pitches of same type
- With constant movement/velocity, higher velocity/movement shoudl score a pitch higher
- Reaction time: calculate time (with extension) for ball to travel to home plate
- Reaction x Movement: how much ball deviates in given reaction time
- Importantly: 105 Stuff+ fastball is not necessarily "better" than 100 Stuff+ curveball
- Aggregated to pitch type

#### Location+: 
- Given pitch type and situation, expected run value of a pitch thrown into a certain area
- Context aware: "groundout" pitch with double play in effect > "groundout" pitch with runner on 3rd and no outs
- Like Stuff+, aggregated to pitch type

#### Pitching+: 
- Try to reward maximized Location+ w/ weight for Stuff+ 
- E.g: high Stuff+ pitch in less-than-ideal location could score equal to a lower Stuff+ pitch in perfect location
- Give score on + scale that shows result

### bestPitch+: 
- Identify, given pitch arsenal, best pitch/general location to throw it in (allowing for "zone," not a pinpoint)
- Given pitcher arsenal (using avg. Stuff+ and Location+ from spot), calculate hypothetical Pitching+ score (bestPitch)
- Identify if pitch "idea" was good (right pitch type/location)
- bestPitch - Pitching+ = bestPitch+ 
- Should always be negative or close to 0

### Planned Workflow
1. Stuff+: Calculate Stuff+ figures for each pitch type
2. Location+: run expectancy change for pitch thrown in location given type and situation
3. Pitching+: identify weight that seems to produce best results (xWOBA? xERA? TBD)
4. bestPitch+: identify discrepancy between Pitching+ and the optimal pitch (scores and run expectancy change)
