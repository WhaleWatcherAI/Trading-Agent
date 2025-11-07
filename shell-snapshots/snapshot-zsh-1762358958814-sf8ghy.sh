# Snapshot file
# Unset all aliases to avoid conflicts with functions
unalias -a 2>/dev/null || true
# Functions
# Shell Options
setopt nohashdirs
setopt login
# Aliases
alias -- jlab='~/start_jupyter.sh'
alias -- jupyter='~/Library/Python/3.9/bin/jupyter'
alias -- run-help=man
alias -- which-command=whence
# Check for rg availability
if ! command -v rg >/dev/null 2>&1; then
  alias rg='/opt/homebrew/Caskroom/claude-code/2.0.31/claude --ripgrep'
fi
export PATH=/Users/coreycosta/.opencode/bin\:/Applications/Docker.app/Contents/Resources/bin\:/usr/local/share/dotnet\:/usr/local/share/dotnet\:/Users/coreycosta/.npm-global/bin\:/opt/homebrew/bin\:/opt/homebrew/sbin\:/Library/Frameworks/Python.framework/Versions/3.13/bin\:/usr/local/bin\:/System/Cryptexes/App/usr/bin\:/usr/bin\:/bin\:/usr/sbin\:/sbin\:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin\:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin\:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin\:/usr/local/share/dotnet\:~/.dotnet/tools\:/Users/coreycosta/Library/Python/3.9/bin
