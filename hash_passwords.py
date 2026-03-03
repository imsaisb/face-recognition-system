import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hash all plain passwords (run this once!)
stauth.Hasher.hash_passwords(config['credentials'])

# Save back to file (now with hashed passwords)
with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print("Passwords hashed and saved to config.yaml. Delete this script after use.")