"""
    SpikenautKelly

Half-Kelly position sizing with SNN confidence → fraction mapping.

Maps a spiking neural network's output confidence [0, 1] to an optimal
capital fraction using the Kelly Criterion, with half-Kelly variance reduction.

## Formula

```
b  = avg_win / avg_loss          # payoff ratio
F* = (p·b − (1−p)) / b          # full Kelly fraction
F  = F* / 2                      # half-Kelly (standard practice)
F  = clamp(F, 0.02, 0.20)        # conservative bounds
```

## Provenance

Extracted from Eagle-Lander, the author's own private neuromorphic GPU supervisor
repository (closed-source). Source files:

- `execution/src/kelly.rs` — Rust implementation of Kelly fraction and half-Kelly
- `execution/market_kelly.jl` — original Julia position sizing with SNN confidence mapping

The Kelly sizing module ran in production for Dynex/Quai/Qubic/BTC portfolio decisions
driven by a 16-neuron LIF SNN before being open-sourced as a standalone Julia package.

No Julia package previously mapped neural confidence scores directly to Kelly-optimal
position fractions — this fills that gap.

## References

- Kelly, J.L. (1956). A New Interpretation of Information Rate.
  *Bell System Technical Journal*, 35(4), 917–926.
  https://doi.org/10.1002/j.1538-7305.1956.tb03809.x

- Thorp, E.O. (1969). Optimal Gambling Systems for Favorable Games.
  *Review of the International Statistical Institute*, 37(3), 273–293.
  Introduced half-Kelly as a practical variance reduction technique.

- MacLean, L.C., Thorp, E.O., & Ziemba, W.T. (2011).
  *The Kelly Capital Growth Investment Criterion*. World Scientific.
  Comprehensive treatment of Kelly variants and implementation considerations.

## Example

```julia
using SpikenautKelly

# From SNN output statistics
f = kelly_fraction(win_rate=0.58, avg_win=8.50, avg_loss=5.20)
println("Trade fraction: \$(round(f * 100, digits=1))%")  # e.g. 8.4%

# Map SNN neuron confidence directly
f = from_confidence(confidence=0.72, payoff_ratio=0.015)
```
"""
module SpikenautKelly

export kelly_fraction, from_confidence, half_kelly, RiskTier, risk_tier, PositionSize, size_position

# ── Core Kelly Formula ────────────────────────────────────────────────────────

"""
    kelly_fraction(; win_rate, avg_win, avg_loss) -> Float64

Compute half-Kelly position fraction from historical trade statistics.

# Arguments
- `win_rate`: fraction of trades that were profitable (0.0–1.0)
- `avg_win`:  average profit of winning trades (same units as `avg_loss`)
- `avg_loss`: average loss of losing trades (absolute value)

# Returns
Half-Kelly fraction clamped to [0.02, 0.20].

# Example
```julia
f = kelly_fraction(win_rate=0.55, avg_win=8.50, avg_loss=5.20)
# → ~0.084 (8.4%)
```
"""
function kelly_fraction(; win_rate::Float64, avg_win::Float64, avg_loss::Float64)::Float64
    win_rate = clamp(win_rate, 0.01, 0.99)
    avg_win  = max(avg_win,  1e-9)
    avg_loss = max(avg_loss, 1e-9)

    b = avg_win / avg_loss          # payoff ratio
    q = 1.0 - win_rate

    full = (win_rate * b - q) / b
    half = full * 0.5               # half-Kelly for variance reduction

    return clamp(half, 0.02, 0.20)
end

"""
    half_kelly(p, b) -> Float64

Kelly fraction given win probability `p` and payoff ratio `b`, halved.

# Example
```julia
f = half_kelly(0.60, 1.5)  # 60% win rate, 1.5x avg payoff
```
"""
function half_kelly(p::Float64, b::Float64)::Float64
    p = clamp(p, 0.01, 0.99)
    b = max(b, 1e-9)
    q = 1.0 - p
    full = (p * b - q) / b
    return clamp(full * 0.5, 0.0, 1.0)
end

# ── SNN Confidence Mapping ────────────────────────────────────────────────────

"""
    from_confidence(; confidence, payoff_ratio=0.01) -> Float64

Map SNN output confidence [0, 1] to a Kelly position fraction.

SNN confidence is treated as `win_probability` with the given `payoff_ratio`
representing the expected price move (e.g. 0.01 = 1% expected move).

# Arguments
- `confidence`:    SNN output in [0, 1] (treat as win probability)
- `payoff_ratio`:  expected move as fraction of price (default 0.01)

# Example
```julia
f = from_confidence(confidence=0.85, payoff_ratio=0.01)
# High-confidence signal → moderate position size
```
"""
function from_confidence(; confidence::Float64, payoff_ratio::Float64=0.01)::Float64
    p = clamp(confidence, 0.01, 0.99)
    b = max(payoff_ratio, 1e-9)
    q = 1.0 - p
    full = (p * b - q) / b
    half = full * 0.5
    return clamp(half, 0.0, 1.0)
end

# ── Risk Tiers ────────────────────────────────────────────────────────────────

"""
    @enum RiskTier

Qualitative risk classification derived from SNN confidence.
"""
@enum RiskTier begin
    Aggressive   = 4    # confidence ≥ 0.95
    Moderate     = 3    # confidence ≥ 0.85
    Conservative = 2    # confidence ≥ 0.70
    Minimal      = 1    # confidence < 0.70
end

"""
    risk_tier(confidence) -> RiskTier

Classify SNN confidence into a qualitative risk tier.
"""
function risk_tier(confidence::Float64)::RiskTier
    if confidence >= 0.95
        return Aggressive
    elseif confidence >= 0.85
        return Moderate
    elseif confidence >= 0.70
        return Conservative
    else
        return Minimal
    end
end

# ── Position Sizing ───────────────────────────────────────────────────────────

"""
    PositionSize

Result of a Kelly position sizing calculation.

# Fields
- `units`:          number of asset units to trade
- `kelly_fraction`: fraction of account balance committed
- `confidence`:     original SNN confidence input
- `risk`:           qualitative risk tier
- `account_risk_pct`: percentage of account balance at risk
"""
struct PositionSize
    units::Float64
    kelly_fraction::Float64
    confidence::Float64
    risk::RiskTier
    account_risk_pct::Float64
end

"""
    size_position(; confidence, price, account_balance,
                    payoff_ratio=0.01, kelly_scalar=0.5) -> PositionSize

Compute full position sizing from SNN confidence.

# Arguments
- `confidence`:      SNN output [0, 1]
- `price`:           current asset price (USD)
- `account_balance`: total account in USD
- `payoff_ratio`:    expected price move fraction (default 1%)
- `kelly_scalar`:    fractional Kelly multiplier (default 0.5 = half-Kelly)

# Example
```julia
pos = size_position(confidence=0.90, price=65_000.0, account_balance=10_000.0)
println("Buy \$(round(pos.units, digits=6)) BTC (\$(round(pos.account_risk_pct, digits=1))% of account)")
```
"""
function size_position(;
    confidence::Float64,
    price::Float64,
    account_balance::Float64,
    payoff_ratio::Float64 = 0.01,
    kelly_scalar::Float64 = 0.5,
)::PositionSize
    p = clamp(confidence, 0.01, 0.99)
    b = max(payoff_ratio, 1e-9)
    q = 1.0 - p
    full_k = (p * b - q) / b
    k = clamp(full_k * kelly_scalar, 0.0, 1.0)

    units = (k * account_balance) / max(price, 1e-9)
    risk_pct = (units * price) / account_balance * 100.0

    return PositionSize(units, k, confidence, risk_tier(confidence), risk_pct)
end

end # module
