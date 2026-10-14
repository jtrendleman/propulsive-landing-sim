# Physics Model

## Sign Convention (binding contract)

+y is UP. The landing pad is at y = 0.

| Quantity | Symbol | Sign rule |
|---|---|---|
| Altitude | y | >= 0 always; y = 0 is touchdown |
| Velocity | v | NEGATIVE when descending, positive when climbing |
| Acceleration | a | positive means v is increasing |
| Gravity | F_g | always negative: F_g = -m*g |
| Thrust | F_T | always positive: F_T = +throttle * T_max |
| Drag | F_D | opposes velocity: F_D = -0.5*rho*Cd*A*v*abs(v) |

Note: during a landing burn, positive acceleration is the goal.
Slowing a descent means v increases, from -80 toward 0.

The v*abs(v) form in drag gives v^2 magnitude with automatic sign
correction, valid for both descent and ascent, with no branching.

All equations, plots, and reward terms in this repo obey this
convention. Any function that treats down as positive is a bug.
