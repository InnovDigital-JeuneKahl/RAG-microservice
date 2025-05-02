import requests

url = "http://127.0.0.1:8000/system/reset"
params = {"confirm": True}

try:
    # Make the request
    response = requests.post(url, params=params)
    
    # Print detailed debug info
    print(f"Status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response content: {response.text}")
    
    # Only try to parse JSON if we got a successful response
    if response.status_code == 200:
        try:
            print(f"JSON response: {response.json()}")
        except requests.exceptions.JSONDecodeError:
            print("Response is not valid JSON")
    else:
        print(f"Request failed with status code {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print(f"Connection failed. Is the server running at {url}?")
except Exception as e:
    print(f"Error: {str(e)}")