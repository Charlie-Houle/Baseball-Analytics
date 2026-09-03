# Baseball-Analytics
Baseball Analytics project storage 

# Pitching+ (pitching_plus)
Model ranking pitches on 100+ scale using its movement profile (Stuff+) and expected run value given location and pitch type (Location+). Stuff+ and Location+ are combined into Pitching+, which is compared against the best score achievable from the pitcher's own arsenal in that same situation (bestPitch) to produce "bestPitch+", a decision-quality metric, not another stuff/location score, to inform pitching strategy.

# Slippery Slope (slippery_slope)
Investigation of how a count shapes the rest of an at-bat, in particular whether a first-pitch ball starts a "slippery slope" toward a walk or a more conservative approach from the pitcher. Starts with baseline count-state outcome rates (strikeout/walk/hit rates by count, both on an immediate-pitch and a resulting-at-bat basis), then extends to pitch-type strategy (a "Pitcher Plinko"-style breakdown of arsenal usage by count) and eventually pitch-location strategy, both per-pitcher and leaguewide.