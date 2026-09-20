import os
import sys
import socket
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_airgap_violation():
    """Validates zero outbound network calls are attempted on runtime by importing app.py which patches socket."""
    
    # Import app.py which applies the socket monkey-patch
    # Note: Streamlit execution is bypassed, we just test the patch itself.
    import app
    
    # Allowed local connection
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # We don't actually need to connect to a real server, just testing the interceptor logic
        # But connecting to an unresolvable port on localhost might raise ConnectionRefusedError 
        # instead of PermissionError, which means the air-gap allows it.
        s.connect(("127.0.0.1", 99999))
    except ConnectionRefusedError:
        pass # Expected
    except OverflowError:
        pass # Expected for port 99999
    except PermissionError:
        pytest.fail("Air-gap erroneously blocked localhost connection.")
    finally:
        s.close()

    # Blocked external connection
    with pytest.raises(PermissionError, match="Air-Gap Violation: Outbound connection to .* blocked."):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
        s.close()

    with pytest.raises(PermissionError, match="Air-Gap Violation: Outbound connection to .* blocked."):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("fonts.googleapis.com", 443))
        s.close()
