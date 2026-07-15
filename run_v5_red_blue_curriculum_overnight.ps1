$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$arguments = @(
    "--seed-team", "v5_red_scrimmage_until8pm_seed20260712_objective_champion_team.npy",
    "--seed-base", "v5_red_scrimmage_until8pm_seed20260712_base.npy",
    "--curriculum", "blue_curriculum_stages.npy",
    "--games", "30",
    "--seasons", "1000000",
    "--elite", "2",
    "--mut", "0.03",
    "--seed", "20260713",
    "--checkpoint-every", "1000",
    "--benchmark-every", "25",
    "--benchmark-trials", "100",
    "--mastery-streak", "3",
    "--max-hours", "8",
    "--out", "v5_red_blue_curriculum_overnight_seed20260713"
)

& .\v5_red_blue_curriculum.exe @arguments `
    1> v5_red_blue_curriculum_overnight_seed20260713.log `
    2> v5_red_blue_curriculum_overnight_seed20260713.err.log
