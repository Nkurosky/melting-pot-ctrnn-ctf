$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$deadline = (Get-Date).Date.AddHours(20)
$remainingHours = ($deadline - (Get-Date)).TotalHours
if ($remainingHours -le 0) {
    throw "Today's 8:00 PM deadline has already passed."
}

$hours = $remainingHours.ToString([Globalization.CultureInfo]::InvariantCulture)
$arguments = @(
    "--pool", "120",
    "--games", "30",
    "--seasons", "1000000",
    "--mut", "0.03",
    "--elite", "4",
    "--seed", "20260712",
    "--red-base", "v4_eventonly_8h_seed20260712_red.npy",
    "--blue-base", "v4_eventonly_8h_seed20260712_blue.npy",
    "--checkpoint-every", "250",
    "--benchmark-every", "10",
    "--benchmark-trials", "100",
    "--max-hours", $hours,
    "--out", "v5_elo4v4_until8pm_seed20260712"
)

& .\v5_elo_4v4.exe @arguments `
    1> v5_elo4v4_until8pm_seed20260712.log `
    2> v5_elo4v4_until8pm_seed20260712.err.log
