import os
import ctypes
import atexit

# Set thread execution state to prevent system from sleeping
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040  # Optional, useful for media apps
ES_DISPLAY_REQUIRED = 0x00000002  # Optional, useful for display on

def prevent_sleep():
    if os.name != 'nt':
        return
    print("Preventing system sleep...")
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED | ES_DISPLAY_REQUIRED 
    )
    atexit.register(allow_sleep)

def allow_sleep():
    if os.name != 'nt':
        return
    print("Allowing system to sleep again...")
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    # Đảm bảo máy không sleep trong quá trình chạy
    prevent_sleep()
    pause = input("Press Enter to allow system sleep again...")  # Wait for user input
    # Khi chương trình kết thúc, cho phép sleep lại
