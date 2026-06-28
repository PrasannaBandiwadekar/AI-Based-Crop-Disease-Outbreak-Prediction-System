import requests
import os

class Notifier:
    """Base category for all notification delivery systems."""
    def send(self, message: str, recipient: str):
        raise NotImplementedError("Subclasses must implement send method")

class Fast2SMSNotifier(Notifier):
    """
    Implementation for Fast2SMS (India-specific).
    Requires a valid API key from fast2sms.com.
    """
    def __init__(self, api_key: str = None):
        # Use env variable if not provided
        self.api_key = api_key or os.getenv("FAST2SMS_API_KEY")
        self.url = "https://www.fast2sms.com/dev/bulkV2"

    def send(self, message: str, recipient: str):
        if not self.api_key:
            print(f"DEBUG [Fast2SMS]: No API Key. Would have sent to {recipient}: {message}")
            return False
            
        payload = {
            "message": message,
            "language": "english",
            "route": "q",
            "numbers": recipient,
        }
        
        headers = {
            'authorization': self.api_key,
            'Content-Type': "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(self.url, data=payload, headers=headers)
            result = response.json()
            if result.get("return"):
                print(f"SMS Sent successfully to {recipient}")
                return True
            else:
                print(f"Failed to send SMS: {result.get('message')}")
                return False
        except Exception as e:
            print(f"Error sending SMS: {str(e)}")
            return False

class AlertManager:
    """Consolidates advisory results and determines delivery."""
    def __init__(self, notifier: Notifier):
        self.notifier = notifier

    def process_advisory(self, advisory_json: dict, phone_number: str):
        status = advisory_json.get("status")
        message = advisory_json.get("advisory")
        
        # Delivery Gating: Only send SMS for CRITICAL alerts
        if status == "CRITICAL":
            return self.notifier.send(message, phone_number)
        else:
            print(f"Internal Log: {status} status - SMS withheld to avoid over-alerting.")
            return None

# Simple Demo Logic
if __name__ == "__main__":
    # Mocking the use case
    mock_advisory = {
        "status": "CRITICAL",
        "advisory": "🔴 CRITICAL: Potential Red Rot explosion predicted. Persistent standing water detected. ACTION: Drain field immediately."
    }
    
    # Using the generic handler
    sms_handler = Fast2SMSNotifier(api_key="MOCK_KEY_FOR_DEMO")
    manager = AlertManager(sms_handler)
    
    print("Processing Sample Advisory...")
    manager.process_advisory(mock_advisory, "9876543210")
