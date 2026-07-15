$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$common = @(
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-static",
    "-static-libgcc",
    "-static-libstdc++"
)

$targets = @(
    @{ Source = "v4.cpp"; Output = "v4.exe" },
    @{ Source = "v5_elo_4v4.cpp"; Output = "v5_elo_4v4.exe" },
    @{ Source = "v5_red_scrimmage.cpp"; Output = "v5_red_scrimmage.exe" },
    @{ Source = "v5_red_blue_curriculum.cpp"; Output = "v5_red_blue_curriculum.exe" }
)

foreach ($target in $targets) {
    Write-Host "Building $($target.Source) -> $($target.Output)"
    & g++ @common -o $target.Output $target.Source
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed for $($target.Source)"
    }
}

Write-Host "All C++ study targets built successfully."
