def calculate_ttr(tokens):
    if len(tokens) == 0:
        return 0.0

    unique_tokens = set()

    for token in tokens:
        unique_tokens.add(token)

    return len(unique_tokens) / len(tokens)


def calculate_mattr(tokens, window_size=50):
    if len(tokens) == 0:
        return 0.0

    if len(tokens) < window_size:
        return calculate_ttr(tokens)

    window_ttr_values = []
    last_window_start = len(tokens) - window_size

    for window_start in range(last_window_start + 1):
        window_end = window_start + window_size
        window_tokens = tokens[window_start:window_end]
        window_ttr = calculate_ttr(window_tokens)
        window_ttr_values.append(window_ttr)

    return sum(window_ttr_values) / len(window_ttr_values)
