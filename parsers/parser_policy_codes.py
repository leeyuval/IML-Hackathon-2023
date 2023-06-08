import pandas as pd
import re

"""
Hackathon 2023
- 
- 
- 
-

"""


def policy_parser(df: pd.DataFrame) -> pd.DataFrame:
    policy_pattern = r"(\d+[DN])"
    no_show_pattern = r"_([NP]\d*)$"

    # Extract the policies using the policy pattern
    policies = df['cancellation_policy_code'].str.findall(policy_pattern)

    # Create a list to store the policy objects
    policy_objects = []

    # Process each policy in the list
    for policy_list in policies:
        policy_object = {}
        for policy in policy_list:
            days = re.findall(r"\d+", policy)[0]
            if 'D' in policy:
                policy_object['days_before'] = int(days)
            elif 'N' in policy:
                policy_object['night_charge'] = int(days)
        policy_objects.append(policy_object)

    # Extract the no-show policy if present
    no_show_policy = df['cancellation_policy_code'].str.extract(no_show_pattern, expand=False)
    if not no_show_policy.isna().values[0]:
        if no_show_policy.values[0][0] == 'P':
            policy_objects[-1]['no_show_policy'] = 'percentage_charge'
            policy_objects[-1]['percentage_charge'] = int(no_show_policy.values[0][1:])
        elif no_show_policy.values[0][0] == 'N':
            policy_objects[-1]['no_show_policy'] = 'night_charge'
            policy_objects[-1]['night_charge'] = int(no_show_policy.values[0][1:])
    return df


if __name__ == '__main__':
    pass
