# helpful script to help find cl.exe and add it to the PATH (was having issues with nvcc not finding cl.exe)
# This script will find the Visual Studio cl.exe compiler and add it to your PATH
# Must be run as Administrator

# First let's check if we're running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script needs to be run as Administrator. Right-click the PowerShell icon and select 'Run as Administrator'."
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# Define possible Visual Studio paths to check
$potentialPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022",
    "C:\Program Files\Microsoft Visual Studio\2019",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019"
)

$clPath = $null

Write-Host "Searching for cl.exe in Visual Studio installations..." -ForegroundColor Cyan

# Search for cl.exe in each potential Visual Studio installation path
foreach ($baseDir in $potentialPaths) {
    if (Test-Path $baseDir) {
        Write-Host "Checking $baseDir..." -ForegroundColor Yellow
        
        # Use Get-ChildItem to find cl.exe recursively
        $clFiles = Get-ChildItem -Path $baseDir -Filter "cl.exe" -Recurse -ErrorAction SilentlyContinue | 
                   Where-Object { $_.FullName -match "\\Hostx64\\x64\\" }
        
        if ($clFiles.Count -gt 0) {
            # We found it - take the first match that's in the Hostx64\x64 directory
            $clPath = $clFiles[0].DirectoryName
            Write-Host "Found cl.exe at: $($clFiles[0].FullName)" -ForegroundColor Green
            break
        }
    }
}

if (-not $clPath) {
    Write-Host "Could not find cl.exe in standard Visual Studio locations." -ForegroundColor Red
    Write-Host "Please make sure Visual Studio is installed with C++ development tools." -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# Get current PATH from the registry
$currentPath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::Machine)

# Check if the path is already in PATH
if ($currentPath -like "*$clPath*") {
    Write-Host "The cl.exe directory is already in your PATH." -ForegroundColor Green
} else {
    # Add the new path to the existing PATH
    $newPath = "$currentPath;$clPath"
    
    # Update the PATH environment variable
    try {
        [Environment]::SetEnvironmentVariable("Path", $newPath, [EnvironmentVariableTarget]::Machine)
        Write-Host "Successfully added cl.exe directory to your system PATH." -ForegroundColor Green
        Write-Host "Added: $clPath" -ForegroundColor Green
    } catch {
        Write-Host "Error updating PATH: $_" -ForegroundColor Red
    }
}

# Test if nvcc can now find cl.exe
Write-Host "`nTesting CUDA compilation capability..." -ForegroundColor Cyan

# Update current session's PATH variable
$env:Path = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::Machine)

# Try to run nvcc to see if it can find cl.exe now
$testOutput = cmd /c nvcc --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "nvcc is working correctly. You should now be able to compile CUDA code." -ForegroundColor Green
    Write-Host "Note: You may need to restart any open command prompts or IDEs for the changes to take effect." -ForegroundColor Yellow
} else {
    Write-Host "nvcc test failed. You may need to restart your computer for the PATH changes to take effect." -ForegroundColor Red
    Write-Host "Error: $testOutput" -ForegroundColor Red
}

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")