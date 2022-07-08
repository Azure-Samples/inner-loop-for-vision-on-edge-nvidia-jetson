#! /bin/bash
export PIPX_HOME=${1:-"/usr/local/py-utils"}
USERNAME=${2:-"automatic"}
UPDATE_RC=${3:-"true"}

DEFAULT_UTILS=("pylint" "flake8" "autopep8" "black" "yapf" "mypy" "pydocstyle" "pycodestyle" "bandit" "pipenv" "virtualenv")

# Determine the appropriate non-root user
if [ "${USERNAME}" = "auto" ] || [ "${USERNAME}" = "automatic" ]; then
    USERNAME=""
    POSSIBLE_USERS=("vscode" "node" "codespace" "$(awk -v val=1000 -F ":" '$3==val{print $1}' /etc/passwd)")
    for CURRENT_USER in "${POSSIBLE_USERS[@]}"; do
        if id -u "${CURRENT_USER}" > /dev/null 2>&1; then
            USERNAME=${CURRENT_USER}
            break
        fi
    done
    if [ "${USERNAME}" = "" ]; then
        USERNAME=root
    fi
elif [ "${USERNAME}" = "none" ] || ! id -u ${USERNAME} > /dev/null 2>&1; then
    USERNAME=root
fi

updaterc() {
    if [ "${UPDATE_RC}" = "true" ]; then
        echo "Updating /etc/bash.bashrc and /etc/zsh/zshrc..."
        if [[ "$(cat /etc/bash.bashrc)" != *"$1"* ]]; then
            echo -e "$1" >> /etc/bash.bashrc
        fi
        if [ -f "/etc/zsh/zshrc" ] && [[ "$(cat /etc/zsh/zshrc)" != *"$1"* ]]; then
            echo -e "$1" >> /etc/zsh/zshrc
        fi
    fi
}

export PIPX_BIN_DIR="${PIPX_HOME}/bin"
export PATH="${PYTHON_INSTALL_PATH}/bin:${PIPX_BIN_DIR}:${PATH}"

# Create pipx group, dir, and set sticky bit
if ! cat < /etc/group | grep -e "^pipx:" > /dev/null 2>&1; then
    groupadd -r pipx
fi
usermod -a -G pipx ${USERNAME}
umask 0002
mkdir -p "${PIPX_BIN_DIR}"
chown :pipx "${PIPX_HOME}" "${PIPX_BIN_DIR}"
chmod g+s "${PIPX_HOME}" "${PIPX_BIN_DIR}"

# Install tools
echo "Installing Python tools..."
export PYTHONUSERBASE=/tmp/pip-tmp
export PIP_CACHE_DIR=/tmp/pip-tmp/cache
pipx_path=""
if ! type pipx > /dev/null 2>&1; then
    pip3 install --disable-pip-version-check --no-cache-dir --user pipx 2>&1
    /tmp/pip-tmp/bin/pipx install --pip-args=--no-cache-dir pipx
    pipx_path="/tmp/pip-tmp/bin/"
fi
for util in "${DEFAULT_UTILS[@]}"; do
    if ! type "${util}" > /dev/null 2>&1; then
        ${pipx_path}pipx install --system-site-packages --pip-args '--no-cache-dir --force-reinstall' "${util}"
    else
        echo "${util} already installed. Skipping."
    fi
done
rm -rf /tmp/pip-tmp

updaterc "$(cat << EOF
export PIPX_HOME="${PIPX_HOME}"
export PIPX_BIN_DIR="${PIPX_BIN_DIR}"
if [[ "\${PATH}" != *"\${PIPX_BIN_DIR}"* ]]; then export PATH="\${PATH}:\${PIPX_BIN_DIR}"; fi
EOF
)"