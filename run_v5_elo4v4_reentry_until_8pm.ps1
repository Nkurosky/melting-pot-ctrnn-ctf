$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$deadline = (Get-Date).Date.AddHours(20)
$remainingHours = ($deadline - (Get-Date)).TotalHours
if ($remainingHours -le 0) {
    throw "Today's 8:00 PM deadline has already passed."
}

$hours = $remainingHours.ToString([Globalization.CultureInfo]::InvariantCulture)
$prefix = "v5_elo4v4_reentry_seed20260713"
$arguments = @(
    "--pool", "120",
    "--games", "30",
    "--seasons", "1000000",
    "--mut", "0.03",
    "--elite", "4",
    "--seed", "20260713",
    "--red-base", "v5_red_blue_curriculum_overnight_seed20260713_base.npy",
    "--blue-base", "champions/blue_elo3000_base_s11750.npy",
    "--checkpoint-every", "250",
    "--benchmark-every", "10",
    "--benchmark-trials", "100",
    "--max-hours", $hours,
    "--out", $prefix
)

& .\v5_elo_4v4.exe @arguments `
    1> "$prefix.log" `
    2> "$prefix.err.log"
