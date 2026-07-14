# One-command build for Windows: locates the MSVC dev environment (if needed),
# then configures and builds all stages via CMake + Ninja.

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if ((Get-Command cl.exe -ErrorAction SilentlyContinue) -eq $null -and (Test-Path $vswhere)) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsPath) {
        $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvars) {
            cmd /c "`"$vcvars`" && set" | ForEach-Object {
                if ($_ -match '^(.*?)=(.*)$') {
                    Set-Item -Path "Env:\$($matches[1])" -Value $matches[2]
                }
            }
        }
    }
}

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build build
