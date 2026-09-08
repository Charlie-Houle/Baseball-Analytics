### Purpose: 
Create "+" style model that scores on following fronts: 
1. Stuff+ (how "filthy" a pitch is)
2. Location+ (given context of AB, was pitch thrown in good location) 
3. Pitching+ (given Stuff+ and Location+, how good was the pitch)
4. bestPitch+ (difference between "optimal" pitch and Pitching+)

### Basis of "Good":

#### Stuff+: 
- A trained model (HistGradientBoostingRegressor per pitch type, 5-fold
  out-of-fold cross-validation) predicting a pitch's expected run value
  (delta_pitcher_run_exp) from its own physics: release speed/spin/
  extension, acceleration, reaction-time-adjusted movement, and a
  seam-shifted-wake proxy (axis_differential: angular gap between measured
  spin axis and observed movement direction)
- Retooled from an original unsupervised PCA-composite design ("outlierness"
  to other pitches of the same type, no outcome data); see docs/dev_log.md
  for the full history of both designs and why the second replaced the first
- Reaction time: calculate time (with extension) for ball to travel to home plate
- Reaction x Movement: how much ball deviates in given reaction time
- Built from a pitch's own physics only. No location, count, or
  situational features (Location+/Pitching+'s job), and no comparison
  against the pitcher's other pitch types (e.g. changeup velocity diff to
  primary fastball)
- Out-of-fold scoring: no pitch's own realized outcome ever leaks into its
  own Stuff+ score
- A 105 Stuff+ fastball is not necessarily "better" than a 100 Stuff+
  curveball. Scored and calibrated separately per pitch type
- Aggregated to pitch type

#### Location+: 
- Given pitch type and situation, expected run value of a pitch thrown into a certain area
- Context aware: "groundout" pitch with double play in effect > "groundout" pitch with runner on 3rd and no outs
- Like Stuff+, aggregated to pitch type

#### Pitching+: 
- A jointly-trained model (HistGradientBoostingRegressor per pitch type,
  5-fold out-of-fold cross-validation) predicting a pitch's expected run
  value directly from its combined physics AND location/count/situational
  features together, not a weighted blend of separately-computed Stuff+ and
  Location+ scores
- Retooled from an original weighted-blend design (an intercept and two
  z-scored slopes on log(Stuff+) and mean Location+ run value); see
  docs/dev_log.md for the full history of both designs and why the second
  replaced the first, including a leave-one-season-out validation
- E.g: high Stuff+ physics in less-than-ideal location could score equal to
  a lower Stuff+ physics in perfect location -- the joint model learns
  these interactions directly rather than combining two independent scores
- Give score on + scale that shows result

### bestPitch+: 
- Identify, given pitch arsenal, best pitch/general location to throw it in (allowing for "zone," not a pinpoint)
- Given pitcher arsenal (using avg. Stuff+ and Location+ from spot), calculate hypothetical Pitching+ score (bestPitch)
- Identify if pitch "idea" was good (right pitch type/location)
- bestPitch - Pitching+ = bestPitch+ 
- bestPitch is a max over real candidate pitches, and the actual pitch is one of those candidates, so bestPitch+ should be positive or close to 0

### Planned Workflow
1. Stuff+: Calculate Stuff+ figures for each pitch type
2. Location+: run expectancy change for pitch thrown in location given type and situation
3. Pitching+: identify weight that seems to produce best results (target is delta_pitcher_run_exp, Statcast's own RE288-consistent per-pitch run expectancy change)
4. bestPitch+: identify discrepancy between Pitching+ and the optimal pitch (scores and run expectancy change)