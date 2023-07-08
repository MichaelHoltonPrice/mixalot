class VarSpec:
    """
    Represents the specifications of a variable in a dataset.

    Each VarSpec object captures the variable name, its type (numerical, categorical, or ordinal), 
    a mapping for categorical variables (if applicable), any missing values, and the order of the variable.

    Args:
        var_name (str): Name of the variable.
        var_type (str): Type of the variable. Must be 'numerical', 'categorical', or 'ordinal'.
        categorical_mapping (list[set], optional): List of sets representing the categories of a categorical or ordinal variable.
                                                   Each set includes strings that map to the respective category. 
                                                   This parameter is required if var_type is 'categorical' or 'ordinal'.
        missing_values (Any, optional): Specifies the representation of missing values in the variable. Default is None.
        column_name (str, optional): Specifies a distinct column name in an input file (by default var_name is assumed)

    Raises:
        ValueError: If var_type is not 'numerical', 'categorical', or 'ordinal'.
        ValueError: If var_type is 'categorical' or 'ordinal' but no categorical_mapping is provided.
        ValueError: If the same string appears in more than one set of categorical_mapping.
    """
 
    def __init__(self,
                 var_name,
                 var_type,
                 categorical_mapping=None,
                 missing_values=None,
                 column_name=None):
        self.var_name = var_name
        self.var_type = var_type.lower()
        self.categorical_mapping = categorical_mapping
        self.missing_values = missing_values
        self.column_name = column_name
        self._validate_spec()

    def _validate_spec(self):
        if self.var_type not in ['numerical', 'categorical', 'ordinal']:
            raise ValueError(f"Invalid 'type' field for variable {self.var_name}. Expected 'numerical', 'categorical', or 'ordinal'")
        if self.var_type in ['categorical', 'ordinal'] and self.categorical_mapping is None:
            raise ValueError(f"Missing 'categorical_mapping' field for variable {self.var_name} of type {self.var_type}")
        if self.var_type in ['categorical', 'ordinal']:
            self._validate_categorical_mapping()

    def _validate_categorical_mapping(self):
        # Check that all entries are sets
        if not all(isinstance(cat_map, set) for cat_map in self.categorical_mapping):
            raise ValueError("All entries in 'categorical_mapping' should be sets")
        # Check for string duplication across different sets
        all_values = [item for cat_map in self.categorical_mapping for item in cat_map]
        if len(all_values) != len(set(all_values)):
            raise ValueError(f"Some strings appear in more than one set for variable {self.var_name}")