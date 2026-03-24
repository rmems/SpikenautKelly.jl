using Test
using SpikenautKelly

@testset "SpikenautKelly" begin
    @testset "kelly_fraction" begin
        f = kelly_fraction(win_rate=0.55, avg_win=8.50, avg_loss=5.20)
        @test 0.02 <= f <= 0.20
        @test f > 0.05  # should be meaningful for edge-positive setup

        # Conservative floor
        f_bad = kelly_fraction(win_rate=0.40, avg_win=1.0, avg_loss=2.0)
        @test f_bad == 0.02  # clamped to floor

        # High win rate → higher fraction
        f_hi = kelly_fraction(win_rate=0.70, avg_win=10.0, avg_loss=5.0)
        f_lo = kelly_fraction(win_rate=0.52, avg_win=10.0, avg_loss=5.0)
        @test f_hi > f_lo
    end

    @testset "from_confidence" begin
        f_high = from_confidence(confidence=0.95, payoff_ratio=0.01)
        f_low  = from_confidence(confidence=0.55, payoff_ratio=0.01)
        @test f_high > f_low
        @test 0.0 <= f_low  <= 1.0
        @test 0.0 <= f_high <= 1.0
    end

    @testset "half_kelly" begin
        f = half_kelly(0.60, 1.5)
        @test 0.0 <= f <= 1.0

        # Half-Kelly should be less than full Kelly
        p, b = 0.60, 1.5
        q = 1.0 - p
        full = (p * b - q) / b
        @test half_kelly(p, b) ≈ full * 0.5 atol=1e-9
    end

    @testset "risk_tier" begin
        @test risk_tier(0.97) == Aggressive
        @test risk_tier(0.90) == Moderate
        @test risk_tier(0.75) == Conservative
        @test risk_tier(0.50) == Minimal
    end

    @testset "size_position" begin
        pos = size_position(confidence=0.90, price=65_000.0, account_balance=10_000.0)
        @test pos.units > 0.0
        @test pos.kelly_fraction > 0.0
        @test pos.risk == Moderate
        @test 0.0 < pos.account_risk_pct < 100.0

        # Higher confidence → larger position
        pos_hi = size_position(confidence=0.95, price=65_000.0, account_balance=10_000.0)
        pos_lo = size_position(confidence=0.72, price=65_000.0, account_balance=10_000.0)
        @test pos_hi.units >= pos_lo.units
    end
end
