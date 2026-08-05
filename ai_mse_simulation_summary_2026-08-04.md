# Simulation of the MSE floor

## Calibration

- mu = 30 percentage points
- sigma_theta = 10 percentage points
- beta = 0.4
- corr(S_j, theta_j) = 0.44
- validation standard error = 3 percentage points
- omega^2 = 80.64 squared percentage points
- floor RMSE = 8.98 percentage points

## Main findings

The fitted log-log slope of excess MSE on m is -1.031. The theoretical 1/m rate corresponds to -1.

The values of m times excess MSE are 40: 200.5, 80: 186.5, 160: 180.7, 320: 185.4, 640: 180.9. Their stability is the direct numerical check of the 1/m approximation.

At m = 40, total MSE is 85.65. The floor is 94.1% of that total, and the shrinking term is 5.01, or 6.2% of the floor.

At m = 320, the shrinking term is 0.579, while omega^2 remains 80.64. The floor is 99.3% of total MSE.

The fitted Lambda/m approximation puts the point at which the shrinking term is below 10% of the floor at approximately m = 25.

The largest relative Monte Carlo standard error among m >= 40 is 2.37%. The largest difference between analytically integrated held-out risk and the simulated J = 1,000 atlas risk is 0.073 MSE units.

## Sensitivity at m = 40

| Score correlation | Validation SE | omega^2 | Excess MSE | Floor share |
|---:|---:|---:|---:|---:|
| 0.30 | 3 | 91.00 | 5.36 | 94.4% |
| 0.44 | 3 | 80.64 | 5.01 | 94.1% |
| 0.70 | 3 | 51.00 | 3.31 | 93.9% |
| 0.44 | 1 | 80.64 | 4.58 | 94.6% |
| 0.44 | 6 | 80.64 | 6.39 | 92.7% |

Changing validation precision changes the shrinking term but not the oracle floor. Changing score quality changes the floor itself.

## Computation

The run used 2,000 Monte Carlo repetitions and finished locally in 5.2 seconds. The structural moment estimator is projected onto sigma_theta^2 >= 0, 0 <= beta <= 1, and sigma_S^2 >= 0. The projection rate is reported in the CSV; it is relevant mainly at the smallest validation budgets and vanishes over the asymptotic range used for the rate check.
