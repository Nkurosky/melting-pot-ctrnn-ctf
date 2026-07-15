$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$deadline = (Get-Date).Date.AddHours(20)
$remainingHours = ($deadline - (Get-Date)).TotalHours
if ($remainingHours -le 0) {
    throw "Today's 8:00 PM deadline has already passed."
}

$hours = $remainingHours.ToString([Globalization.CultureInfo]::InvariantCulture)
$arguments = @(
    "--seed-team", "champions\red_scrimmage_active_seed_team_s9250.npy",
    "--games", "30",
    "--seasons", "1000000",
    "--elite", "2",
    "--mut", "0.03",
    "--seed", "20260712",
    "--checkpoint-every", "1000",
    "--benchmark-every", "25",
    "--benchmark-trials", "100",
    "--max-hours", $hours,
    "--out", "v5_red_scrimmage_until8pm_seed20260712"
)

& .\v5_red_scrimmage.exe @arguments `
    1> v5_red_scrimmage_until8pm_seed20260712.log `
    2> v5_red_scrimmage_until8pm_seed20260712.err.log
