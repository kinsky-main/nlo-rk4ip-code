
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

$pythonExamples = @(
    "conservation_checks_rk4ip.py",
    "fixed_step_soliton_convergence_rk4ip.py",
    "linear_drift_rk4ip.py",
    "raman_scattering_rk4ip.py",
    "runtime_callable_operator_rk4ip.py",
    "runtime_temporal_demo.py",
    "second_order_soliton_rk4ip.py",
    "spm_rk4ip.py",
    "sqlite_snapshot_chunking_demo.py",
    "two_mode_linear_beating_rk4ip.py"
)

foreach ($example in $pythonExamples) {
    python "$repoRoot\examples\python\$example" --replot
}