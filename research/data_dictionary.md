# TouchlineLM Data Dictionary

**Domain**: Football Intelligence & Tactical Event Analytics  
**Primary Schema**: StatsBomb Open Data Standard (Augmented for LM Tokenization)  
**Version**: 1.0 (Phase 1 Baseline)  

---

## 1. Entity Hierarchy & Relational Schema

```
Competition (competition_id, country_name, competition_name)
  └── Season (season_id, season_name)
        └── Match (match_id, match_date, home_team, away_team, home_score, away_score)
              ├── Lineups (team_id, team_name, lineup: [player_id, player_name, jersey_number, positions])
              └── Events [1..N] (id, index, period, timestamp, minute, second, type, team, player, location, ...)
```

---

## 2. Pitch Geometry & Coordinate Systems

StatsBomb uses a right-handed Cartesian coordinate system where the attacking direction is always left-to-right (`x: 0 -> 120`).

- **Length (`x`)**: `0.0` to `120.0` yards.
  - Own Goal Line: `x = 0.0`
  - Defensive Third: `0.0 <= x < 40.0`
  - Middle Third: `40.0 <= x < 80.0`
  - Attacking Third: `80.0 <= x <= 120.0`
  - Opponent Penalty Box: `x >= 102.0` and `18.0 <= y <= 62.0`
  - Opponent Goal Line & Center of Goal: `x = 120.0`, `y = 40.0`
- **Width (`y`)**: `0.0` to `80.0` yards.
  - Right Touchline (from team's attacking view): `y = 80.0`
  - Left Touchline (from team's attacking view): `y = 0.0`
  - Pitch Center: `(x=60.0, y=40.0)`
  - Vertical Channels:
    - Left Wing: `0.0 <= y < 18.0`
    - Left Half-Space: `18.0 <= y < 30.0`
    - Central Corridor: `30.0 <= y <= 50.0`
    - Right Half-Space: `50.0 < y <= 62.0`
    - Right Wing: `62.0 < y <= 80.0`
- **Height (`z`)**: Provided for shots on target `(z=0.0 to ~3.0 yards)`.

---

## 3. Core Event Types and Sub-Attribute Specifications

### 3.1 Event Base Schema
Every event in the sequence contains the following fundamental fields:

| Field Name | Type | Allowed / Valid Range | Description |
|---|---|---|---|
| `id` | UUID | Valid UUID string | Unique event identifier. |
| `index` | Integer | `1` to `N` (~4500) | Monotonically increasing event index within the match. |
| `period` | Integer | `1, 2, 3, 4, 5` | `1`: 1st Half, `2`: 2nd Half, `3`: Extra Time 1, `4`: Extra Time 2, `5`: Penalties. |
| `timestamp` | String | `HH:MM:SS.mmm` | Microsecond timestamp relative to half kickoff. |
| `minute` | Integer | `0` to `120+` | Match minute. |
| `second` | Integer | `0` to `59` | Match second. |
| `type` | Object | Nested `{id, name}` | High-level event category (e.g., Pass, Shot, Carry, Duel). |
| `possession` | Integer | `1` to `K` | Continuous ball possession sequence index. |
| `possession_team` | Object | Nested `{id, name}` | Team currently in possession of the sequence. |
| `team` | Object | Nested `{id, name}` | Team executing the action. |
| `player` | Object | Nested `{id, name}` | Player executing the action (omitted on referee stoppages). |
| `location` | Float Array | `[x, y]` where `x ∈ [0, 120]`, `y ∈ [0, 80]` | Pitch coordinate of origin. |
| `under_pressure` | Boolean | `True`, `None` | Indicator whether active opponent pressure was applied during action. |

---

### 3.2 Action Category: Passing (`type.name == 'Pass'`)

| Field Name | Type | Description & Domain Values |
|---|---|---|
| `pass.recipient` | Object `{id, name}` | Intended receiving teammate. |
| `pass.length` | Float (yards) | Euclidean distance of pass `sqrt((x2-x1)^2 + (y2-y1)^2)`. |
| `pass.angle` | Float (radians) | Directional trajectory angle `[-π, π]`. `0` = straight forward. |
| `pass.height` | Object `{id, name}` | Ground Pass (`Ground`), Low Pass (`Low Pass`), High/Lofted (`High Pass`). |
| `pass.end_location`| Float Array `[x, y]` | Coordinates where pass was received or intercepted. |
| `pass.type` | Object `{id, name}` | Corner, Free Kick, Throw-in, Kick Off, Goal Kick, Recovery. |
| `pass.outcome` | Object `{id, name}` | `Incomplete`, `Out`, `Pass Offside`, `Unknown`. If `None`, pass was successful. |
| `pass.switch` | Boolean | True if pass changes play direction by >35 yards across pitch width. |
| `pass.cross` | Boolean | True if delivered into penalty box from wide channels. |
| `pass.through_ball` | Boolean | True if pass splits defensive backline into open space. |
| `pass.assisted_shot_id` | UUID | Links directly to the shot event created by this key pass (xA generator). |

---

### 3.3 Action Category: Ball Progression (`type.name == 'Carry'`)

| Field Name | Type | Description |
|---|---|---|
| `carry.end_location` | Float Array `[x, y]` | Final coordinates of the ball carry. |
| `duration` | Float (seconds) | Time elapsed during the continuous ball progression. |

**Progressive Action Definition (Industry Analytics Standard)**:
An action is classified as **Progressive** if:
1. Origin and destination are in the opponent's half or defensive/middle boundary.
2. The distance to the center of the opponent's goal `(120, 40)` is reduced by at least 25% relative to the origin distance, OR the ball enters the penalty box.
- Progressive Pass: Pass meeting above criteria and completed.
- Progressive Carry: Carry traveling at least 10 yards forward towards the goal line.

---

### 3.4 Action Category: Goal Threats & Shooting (`type.name == 'Shot'`)

| Field Name | Type | Description & Values |
|---|---|---|
| `shot.statsbomb_xg` | Float `[0.0000, 1.0000]` | Pre-shot probability of a shot resulting in a goal based on historical machine learning models conditioning on distance, angle, defenders between ball and goal, goalkeeper positioning, and pass type. |
| `shot.outcome` | Object `{id, name}` | `Goal`, `Saved`, `Blocked`, `Off Target`, `Post`, `Wayward`, `Saved Off Target`. |
| `shot.technique` | Object `{id, name}` | `Normal`, `Volley`, `Half Volley`, `Lob`, `Overhead Kick`, `Backheel`. |
| `shot.body_part` | Object `{id, name}` | `Right Foot`, `Left Foot`, `Head`, `Other`. |
| `shot.type` | Object `{id, name}` | `Open Play`, `Penalty`, `Direct Free Kick`, `Corner`. |
| `shot.end_location`| Float Array `[x, y, z]`| Pitch end location and height across the goal plane. |
| `shot.first_time` | Boolean | True if struck with a single touch without prior control. |
| `shot.freeze_frame`| List of Objects | Spatial coordinates and roles of all players and goalkeeper at the instant of the strike. |

---

### 3.5 Action Category: Defensive Micro-Actions

| Event Type Name | Key Sub-Attributes | Tactical Significance |
|---|---|---|
| `Duel` | `duel.type.name` (Tackle, Aerial Lost), `duel.outcome.name` (Won, Lost, Success) | Direct 1v1 ground or aerial defensive contest. |
| `Interception` | `interception.outcome.name` (Won, Lost Out, Success) | Disruption of an opponent pass trajectory. |
| `Clearance` | `clearance.body_part`, `clearance.aerial` | Emergency clearing of danger from defensive territory. |
| `Block` | `block.deflection`, `block.offensive` | Physical redirection of opponent pass or shot. |
| `Ball Recovery` | `ball_recovery.offensive`, `ball_recovery.recovery_failure` | Securing a loose ball after a contest or incomplete pass. |
| `Pressure` | `duration`, `counterpress` (Boolean) | Applying physical pressure to the ball carrier within 3 yards. |

---

### 3.6 Action Category: Tactical Alignments

| Event Type Name | Attributes | Description |
|---|---|---|
| `Starting XI` | `tactics.formation` (e.g., 433, 4231, 352), `tactics.lineup` (List of `{player, position}`) | Initial team tactical shape at minute 0:00. |
| `Tactical Shift` | `tactics.formation`, minute, second | Mid-match structural transformation initiated by manager. |
| `Substitution` | `substitution.replacement` `{id, name}` | Tactical substitution of personnel. |
