def format_text(text):
    """Format text for display or processing."""
    return text.strip().capitalize()

def handle_api_error(response):
    """Check for API errors and raise exceptions if necessary."""
    if response.status_code != 200:
        raise Exception(f"API request failed with status code: {response.status_code}")

def load_config(config_file):
    """Load configuration settings from a file."""
    import json
    with open(config_file, 'r') as file:
        return json.load(file)

def save_config(config_file, config_data):
    """Save configuration settings to a file."""
    import json
    with open(config_file, 'w') as file:
        json.dump(config_data, file, indent=4)