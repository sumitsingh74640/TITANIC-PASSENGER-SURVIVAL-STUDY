# Findings

## Demographic factors
- **Sex** is the single strongest predictor. Female survival rate ≈ 74%, male ≈ 19%.
- **Passenger class** is second. 1st ≈ 63%, 2nd ≈ 47%, 3rd ≈ 24%.
- **Age** matters at the extremes: children under 12 survived at ≈ 58%.

## Family structure
- Passengers travelling alone (`IsAlone = 1`) fared worse than those in small families of 2–4.
- Large families (5+) had the lowest survival rate, likely due to coordination difficulty during boarding.

## Model performance
The three models converge to ~80–82% test accuracy, suggesting the dataset ceiling is close to that mark without richer features (Name titles, Cabin deck, Ticket clustering).
