install: requirements.txt
	# Check if pip3 exists, if not, install python3-pip on Debian-based systems; so run the target with sudo in that case.
	which pip3 || (apt update && apt install -y python3-pip)
	# on nixOS the above fails and python312Full package + `python -m venv .venv` is to be used instead.
	which virtualenv || pip3 install virtualenv
	if [ ! -d ".venv" ]; then  virtualenv .venv; fi
	. .venv/bin/activate && pip3 install -r requirements.txt

