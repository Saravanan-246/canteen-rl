def calculate_efficiency(queue, counters):
    base = 100
    queue_penalty = queue * 2
    counter_bonus = counters * 5

    efficiency = base - queue_penalty + counter_bonus

    # Clamp between 0 and 100 (realistic range)
    return max(0, min(100, round(efficiency, 2)))