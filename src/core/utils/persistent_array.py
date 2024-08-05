import json 
import os

class PersistentClasses():
    """
    A class to manage a persistent list of classes, saving and loading from a JSON file

    Attributes:
        filename (str): The name of the file to save/load the list
        array (list): The list of items loaded from the file

    """
    def __init__(self, filename):
        """
        Initializes PersistentClasses with a filename to load/save the list

        Args:
            filename (str): The name of the file to load/save the list 
        """
        self.filename = filename
        self.array = self.load_array()

    def save_array(self):
        """
        Saves the current list of items to the file
        """
        with open(self.filename, 'w') as f:
            json.dump(self.array, f)

    def load_array(self):
        """
        Loads the list of items from the JSON file, if it exists else it returns an empty array

        Returns:
            list: The list of items loaded from the file or an empty list if the file doesn't exist
        """
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f)
        else:
            return []
    
    def clear(self):
        """
        Clears the current list of items and saves an empty list to the JSON file
        """
        self.array = []
        self.save()
    
    def add(self, items):
        """
        Adds a list of items to the current list and saves the updates list to the JSON file

        Args:
            items (list): A list of items to add to the current list
        """
        for item in items:
            self.array.append(item)
        self.save()
    
    def size(self):
        """
        Returns the number of items in the list

        Returns:
            int: The size of the current list
        """
        return len(self.array)
    
    def get_array(self):
        """
        Returns the current list of items as a tuple

        Returns:
            tuple: The current list of items
        """
        return tuple(self.array)