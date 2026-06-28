def compute_fusion_risk(rf_prob, gru_prob):
    """
    Unified Fusion Decision Engine.
    Combines stable risk (RF) and sudden spike detection (GRU).
    """
    
    # 🔴 Logic Rules:
    # RF > 0.6 & GRU > 0.6 -> CRITICAL
    # RF > 0.4 & GRU > 0.6 -> EARLY WARNING (Spike detected on moderate risk)
    # RF > 0.4 & GRU < 0.6 -> MONITOR (Moderate risk, no spike)
    # RF < 0.4 & GRU > 0.6 -> SUSPICIOUS SPIKE (Potential false alarm or very early signal)
    # Else -> SAFE
    
    if rf_prob >= 0.6 and gru_prob >= 0.6:
        return "CRITICAL ALERT", "High stable risk + sudden temporal spike detected."
    elif rf_prob >= 0.4 and gru_prob >= 0.6:
        return "EARLY WARNING", "Moderate stable risk with a significant temporal spike."
    elif rf_prob >= 0.4:
        return "MONITOR", "Stable risk is moderate; keep a close watch."
    elif gru_prob >= 0.6:
        return "SUSPICIOUS SPIKE", "Sudden spike detected despite low stable risk. Inspect for very early symptoms."
    else:
        return "SAFE", "Both stability and spike metrics are within safe limits."

def get_alert_color(status):
    colors = {
        "CRITICAL ALERT": "RED",
        "EARLY WARNING": "ORANGE",
        "MONITOR": "YELLOW",
        "SUSPICIOUS SPIKE": "PURPLE",
        "SAFE": "GREEN"
    }
    return colors.get(status, "GREY")
