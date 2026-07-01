# Installing and Running SRIM 2013 on Kubuntu/Linux with Wine

This guide installs **SRIM 2013** on Linux using **Wine** and places the SRIM files in:
```text
$HOME/Programs/SRIM2013/
```
The procedure is intended for Kubuntu/Ubuntu-like systems using modern Wine.


## 1 Install Wine and helper tools

Enable 32-bit package support and install Wine:
```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install --install-recommends wine wine32 wine64 winetricks xdotool cabextract wget
```
Check that Wine is available:
```bash
wine --version
```

## 2 Define the SRIM installation directory and Wine prefix

Use a dedicated Wine prefix for SRIM:
```bash
export SRIM_DIR="$HOME/Programs/SRIM2013"
export WINEPREFIX="$HOME/.wine-srim2013"
```
Do **not** set `WINEARCH=win32` on modern Wine builds that use WoW64 mode. If it was set previously, unset it:
```bash
unset WINEARCH
```
Create the SRIM directory:
```bash
mkdir -p "$SRIM_DIR"
```
Initialise the Wine prefix:
```bash
wineboot --init
```
If Wine reports something like:
```text
wine: WINEARCH is set to 'win32' but this is not supported in wow64 mode.
```
then remove the prefix and recreate it without `WINEARCH`:
```bash
rm -rf "$WINEPREFIX"
unset WINEARCH
wineboot --init
```

## 3 Download SRIM 2013

Download the SRIM 2013 installer/extractor into the target directory:
```bash
cd "$SRIM_DIR"

wget -O SRIM-2013-Std.exe "http://www.srim.org/SRIM/SRIM-2013-Std.e"
```
The file is downloaded from SRIM with a non-standard `.e` extension, but it is a Windows executable. Saving it as `.exe` allows Wine to run it normally.

## 4 Extract/install SRIM

Run the SRIM installer/extractor:
```bash
cd "$SRIM_DIR"
wine SRIM-2013-Std.exe
```
When the extractor asks where to place the SRIM files, choose:
```text
$HOME/Programs/SRIM2013/
```
After extraction, the directory should contain files such as:
```text
TRIM.exe
SRIM.exe
SRIM-Setup/
```
Check this with:
```bash
ls "$SRIM_DIR"
```

## 5 Install the old Windows run-time files required by SRIM

SRIM/TRIM needs old Microsoft Visual Basic run-time files. If these are missing, running `TRIM.exe` may fail with an error such as:
```text
Library MSVBVM50.DLL ... not found
```
Install the bundled Visual Basic run-time:
```bash
cd "$SRIM_DIR"
wine "SRIM-Setup/MSVBvm50.exe"
```
Copy the bundled OCX files into the SRIM root directory:
```bash
cd "$SRIM_DIR"
cp -n SRIM-Setup/*.ocx . 2>/dev/null || true
```
Register the OCX files with Wine:
```bash
cd "$SRIM_DIR"

for f in "$SRIM_DIR"/*.ocx "$SRIM_DIR"/SRIM-Setup/*.ocx; do
    [ -f "$f" ] || continue
    wine regsvr32 "$(winepath -w "$f")" || true
done
```
Some registration commands may print warnings. Continue as long as `TRIM.exe` starts afterwards.

## 6 Set Wine decimal separators

On systems using European locales, Wine may expose comma decimal separators. SRIM expects dot decimals. Set Wine’s decimal separator to `.`:
```bash
wine reg add "HKEY_CURRENT_USER\Control Panel\International" /v sDecimal /t REG_SZ /d "." /f
wine reg add "HKEY_CURRENT_USER\Control Panel\International" /v sThousand /t REG_SZ /d "," /f
```

## 7 Run SRIM or TRIM

To launch the main SRIM interface:

```bash
cd "$SRIM_DIR"
wine SRIM.exe
```
To launch TRIM directly:
```bash
cd "$SRIM_DIR"
wine TRIM.exe
```
On modern Wine, you may see a message like:
```text
starting ... in experimental wow64 mode
```
This is not necessarily an error. If the SRIM/TRIM window opens, the installation is working.

## 8 Optional: make the environment persistent

To avoid typing the variables every time, add them to `~/.bashrc`:
```bash
echo 'export SRIM_DIR="$HOME/Programs/SRIM2013"' >> ~/.bashrc
echo 'export WINEPREFIX="$HOME/.wine-srim2013"' >> ~/.bashrc
```
Reload the shell configuration:
```bash
source ~/.bashrc
```
Then SRIM can be started with:
```bash
cd "$SRIM_DIR"
wine SRIM.exe
```
or:
```bash
cd "$SRIM_DIR"
wine TRIM.exe
```

