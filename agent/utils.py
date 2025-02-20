def key_map(dictionary: dict, key, default=None):
    """
    Retrieve the value from 'dictionary' for the given 'key'.
    
    Parameters:
        dictionary (dict): The dictionary to search.
        key: The key to look for.
        default: Value to return if the key isn't found (defaults to None).
        
    Returns:
        The value associated with 'key' if it exists, otherwise 'default'.
    """
    return dictionary.get(key, default)