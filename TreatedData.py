workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
             'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
education = ['Bachelors', 'Some-college', 'HS-grad', 'Prof-school',
             'Assoc-acdm', 'Assoc-voc', 'Masters', 'Doctorate', 'School']
maritalStatus = ['Married-civ-spouse', 'Divorced', 'Never-married',
                 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
              'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
relationship = ['Wife', 'Own-child', 'Husband',
                'Not-in-family', 'Other-relative', 'Unmarried']
race = ['White', 'Other']
sex = ['Female', 'Male']
nativeCountry = ['United-States', 'Other']


def getIndex(array, value):
    for i, name in enumerate(array):
        if name == value.strip():
            return i
    return -1


def indexConvert(X):
    for i, values in enumerate(X):
        X[i][1] = getIndex(workclass, values[1])  # Workclass
        X[i][2] = getIndex(education, values[2])  # Education
        X[i][3] = getIndex(maritalStatus, values[3])  # Marital Status
        X[i][4] = getIndex(occupation, values[4])  # Occupation
        X[i][5] = getIndex(relationship, values[5])  # Relationship
        X[i][6] = getIndex(race, values[6])  # Race
        X[i][7] = getIndex(sex, values[7])  # Sex
        X[i][11] = getIndex(nativeCountry, values[11])  # native country
    return X
