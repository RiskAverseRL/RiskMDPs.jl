using RiskMDPs
using Test
@testset "Utility - pw_const_near" begin
    a = [1.0, 2.0, 3.0]
    a = RiskMDPs.Discretized{Float64}(a, a)
    @test RiskMDPs.pw_const_near(a, 0.0) == 1.0
    @test RiskMDPs.pw_const_near(a, 0.25) == 1.0
    @test RiskMDPs.pw_const_near(a, 0.5) == 1.0
    @test RiskMDPs.pw_const_near(a, 0.75) == 1.0
    @test RiskMDPs.pw_const_near(a, 1.0) == 1.0
    @test RiskMDPs.pw_const_near(a, 1.25) == 1.0
    @test RiskMDPs.pw_const_near(a, 1.5) == 1.0
    @test RiskMDPs.pw_const_near(a, 1.75) == 2.0
    @test RiskMDPs.pw_const_near(a, 2.0) == 2.0
    @test RiskMDPs.pw_const_near(a, 2.25) == 2.0
    @test RiskMDPs.pw_const_near(a, 2.5) == 2.0
    @test RiskMDPs.pw_const_near(a, 2.75) == 3.0
    @test RiskMDPs.pw_const_near(a, 3.0) == 3.0
    @test RiskMDPs.pw_const_near(a, 3.25) == 3.0
    @test RiskMDPs.pw_const_near(a, 3.5) == 3.0
    @test RiskMDPs.pw_const_near(a, 3.75) == 3.0
    a = [-1.0, -0.5, 0.0, 0.5, 1.0]
    a = RiskMDPs.Discretized{Float64}(a, a)
    @test RiskMDPs.pw_const_near(a, -1.0) == -1.0
    @test RiskMDPs.pw_const_near(a, -0.75) == -1.0
    @test RiskMDPs.pw_const_near(a, -0.5) == -0.5
    @test RiskMDPs.pw_const_near(a, -0.35) == -0.5
    @test RiskMDPs.pw_const_near(a, -0.25) == -0.5
    @test RiskMDPs.pw_const_near(a, -0.11) == 0.0
    @test RiskMDPs.pw_const_near(a, 0.0) == 0.0
    @test RiskMDPs.pw_const_near(a, 0.25) == 0.0
    @test RiskMDPs.pw_const_near(a, 0.5) == 0.5
    @test RiskMDPs.pw_const_near(a, 0.75) == 0.5
    @test RiskMDPs.pw_const_near(a, 1.0) == 1.0
    @test RiskMDPs.pw_const_near(a, 1.25) == 1.0
    @test RiskMDPs.pw_const_near(a, 1.5) == 1.0
end
