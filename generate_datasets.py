#!/usr/bin/env python3
"""
Synthetic Fundraising Data Generator
Community Impact Foundation - Training Datasets

Generates realistic charity fundraising datasets for Power BI walkthroughs.
All data is synthetic - no real donor information included.

Usage:
    python generate_datasets.py

Output:
    - donors.csv (Walkthrough 1)
    - campaigns.csv (Walkthrough 1)
    - donations.csv (Walkthrough 1)
    - calendar.csv (Both walkthroughs)
    - crm_donors.csv (Walkthrough 2)
    - digital_engagement.csv (Walkthrough 2)
    - financial_transactions.csv (Walkthrough 2)
    - legacy_donations.csv (Walkthrough 2)
    - donor_contact_history.csv (Walkthrough 2)
    - external_wealth_screening.csv (Walkthrough 2)
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize
fake = Faker('en_GB')
np.random.seed(42)  # Reproducibility
random.seed(42)

print("=" * 70)
print("Synthetic Fundraising Data Generator")
print("Community Impact Foundation - Training Datasets")
print("=" * 70)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_DONORS = 6500
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31)
ANALYSIS_START = datetime(2023, 1, 1)
ANALYSIS_END = datetime(2024, 12, 31)

# =============================================================================
# DATASET 1: DONORS (WALKTHROUGH 1)
# =============================================================================

print("[1/10] Generating donors.csv...")

donors = []
channels = ['Online', 'Direct Mail', 'Event', 'Telemarkating', 'Legacy']
channel_weights = [0.35, 0.25, 0.20, 0.15, 0.05]

for i in range(NUM_DONORS):
    donor_id = 1000 + i
    first_name = fake.first_name()
    last_name = fake.last_name()
    
    # Email - 10% invalid/missing
    if random.random() < 0.10:
        email = "" if random.random() < 0.5 else f"{first_name.lower()}.{last_name.lower()}"
    else:
        domains = ['gmail.com', 'outlook.com', 'yahoo.co.uk', 'hotmail.co.uk', 'btinternet.com']
        email = f"{first_name.lower()}.{last_name.lower()}@{random.choice(domains)}"
    
    # Phone - 15% missing
    phone = "" if random.random() < 0.15 else fake.phone_number()
    
    # Address
    address_line1 = fake.street_address()
    city_pool = ['London'] * 20 + ['Birmingham'] * 8 + ['Manchester'] * 7 + \
                ['Leeds'] * 5 + ['Liverpool'] * 5 + ['Bristol'] * 4 + \
                ['Sheffield'] * 3 + ['Newcastle'] * 3 + ['Nottingham'] * 3
    city_pool += [fake.city() for _ in range(42)]
    city = random.choice(city_pool)
    
    # Postcode - 5% invalid
    postcode = fake.bothify(text='??## #??').upper() if random.random() < 0.05 else fake.postcode()
    
    # Acquisition date - weighted toward recent years
    year_weights = [0.10, 0.15, 0.20, 0.25, 0.30]
    acquisition_year = np.random.choice([2020, 2021, 2022, 2023, 2024], p=year_weights)
    acquisition_date = fake.date_between(
        start_date=datetime(acquisition_year, 1, 1),
        end_date=min(datetime(acquisition_year, 12, 31), END_DATE)
    )
    
    acquisition_channel = np.random.choice(channels, p=channel_weights)
    
    # Communication preference
    comm_prefs = ['Email', 'Post', 'Phone', 'None']
    comm_weights = [0.50, 0.30, 0.15, 0.05]
    communication_preference = np.random.choice(comm_prefs, p=comm_weights)
    
    donors.append({
        'donor_id': donor_id,
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'phone': phone,
        'address_line1': address_line1,
        'city': city,
        'postcode': postcode,
        'acquisition_date': acquisition_date,
        'acquisition_channel': acquisition_channel,
        'donor_status': 'Active',  # Will update after donations
        'communication_preference': communication_preference
    })

donors_df = pd.DataFrame(donors)
print(f"    ✓ Generated {len(donors_df):,} donor records")

# =============================================================================
# DATASET 2: CAMPAIGNS (WALKTHROUGH 1)
# =============================================================================

print("[2/10] Generating campaigns.csv...")

campaigns = []
campaign_types = {
    'Emergency Appeal': {'count': 8, 'target_range': (50000, 150000), 'duration_range': (14, 45)},
    'Regular Giving': {'count': 6, 'target_range': (30000, 80000), 'duration_range': (60, 120)},
    'Water Projects': {'count': 5, 'target_range': (25000, 75000), 'duration_range': (30, 90)},
    'Education': {'count': 5, 'target_range': (20000, 60000), 'duration_range': (30, 90)},
    'Health': {'count': 4, 'target_range': (20000, 60000), 'duration_range': (30, 90)},
    'Seasonal': {'count': 4, 'target_range': (15000, 50000), 'duration_range': (14, 30)},
    'Events': {'count': 3, 'target_range': (10000, 30000), 'duration_range': (1, 3)}
}

campaign_id = 100
campaign_start = ANALYSIS_START

for campaign_type, specs in campaign_types.items():
    for i in range(specs['count']):
        if campaign_type == 'Emergency Appeal':
            names = ['Syria Earthquake', 'Gaza Crisis', 'Pakistan Floods', 'Turkey Relief',
                    'Yemen Famine', 'Somalia Drought', 'Bangladesh Floods', 'Sudan Conflict']
            campaign_name = f"{names[i % len(names)]} Emergency Appeal"
        elif campaign_type == 'Seasonal':
            seasons = ['Ramadan', 'Winter', 'Eid', 'Year-End']
            year = 2023 if i < 2 else 2024
            campaign_name = f"{seasons[i % 4]} {year}"
        elif campaign_type == 'Events':
            events = ['Marathon Fundraiser', 'Charity Gala', 'Community Fun Run']
            campaign_name = events[i % 3]
        else:
            campaign_name = f"{campaign_type} Campaign {2023 + i // 3}"
        
        days_offset = random.randint(0, 700)
        start_date = campaign_start + timedelta(days=days_offset)
        duration_days = random.randint(*specs['duration_range'])
        end_date = start_date + timedelta(days=duration_days)
        target_amount = random.randint(*specs['target_range'])
        
        if end_date < datetime(2024, 12, 1):
            campaign_status = 'Completed'
        elif start_date > datetime(2024, 12, 1):
            campaign_status = 'Planned'
        else:
            campaign_status = 'Active'
        
        campaigns.append({
            'campaign_id': campaign_id,
            'campaign_name': campaign_name,
            'campaign_type': campaign_type,
            'start_date': start_date,
            'end_date': end_date,
            'target_amount': target_amount,
            'campaign_status': campaign_status
        })
        campaign_id += 1

campaigns_df = pd.DataFrame(campaigns)
print(f"    ✓ Generated {len(campaigns_df)} campaigns")

# =============================================================================
# DATASET 3: DONATIONS (WALKTHROUGH 1)
# =============================================================================

print("[3/10] Generating donations.csv...")

donations = []
donation_id = 10000

# Assign donor behavior profiles
donor_profiles = {}
for donor_id in donors_df['donor_id']:
    segment = np.random.choice(
        ['Champion', 'Loyal', 'Occasional', 'One-time', 'Lapsed'],
        p=[0.08, 0.15, 0.35, 0.30, 0.12]
    )
    
    donor_profiles[donor_id] = {
        'segment': segment,
        'avg_gift': np.random.lognormal(3.5, 1.2) if segment in ['Champion', 'Loyal'] else np.random.lognormal(2.8, 0.8),
        'frequency_per_year': {
            'Champion': random.randint(8, 24),
            'Loyal': random.randint(4, 8),
            'Occasional': random.randint(2, 4),
            'One-time': 1,
            'Lapsed': 0
        }[segment],
        'regular_giver': random.random() < 0.45 if segment in ['Champion', 'Loyal'] else False
    }

# Generate donations
for donor_id, profile in donor_profiles.items():
    donor_record = donors_df[donors_df['donor_id'] == donor_id].iloc[0]
    acquisition_date = pd.to_datetime(donor_record['acquisition_date'])
    
    if acquisition_date > ANALYSIS_END:
        continue
    
    years_active = min(2, (ANALYSIS_END - acquisition_date).days / 365)
    num_gifts = int(profile['frequency_per_year'] * years_active)
    
    if num_gifts == 0:
        continue
    
    for gift_num in range(num_gifts):
        # 70% tied to campaigns
        if random.random() < 0.7:
            eligible_campaigns = campaigns_df[
                (campaigns_df['start_date'] >= acquisition_date) &
                (campaigns_df['start_date'] <= ANALYSIS_END)
            ]
            if len(eligible_campaigns) > 0:
                campaign = eligible_campaigns.sample(1).iloc[0]
                campaign_id = campaign['campaign_id']
                donation_date = campaign['start_date'] + timedelta(
                    days=random.randint(0, min(30, (campaign['end_date'] - campaign['start_date']).days))
                )
            else:
                campaign_id = None
                donation_date = acquisition_date + timedelta(days=random.randint(0, 700))
        else:
            campaign_id = None
            donation_date = acquisition_date + timedelta(days=random.randint(0, 700))
        
        if donation_date > ANALYSIS_END or donation_date < ANALYSIS_START:
            continue
        
        amount = max(5, round(np.random.normal(profile['avg_gift'], profile['avg_gift'] * 0.3), 2))
        
        if profile['regular_giver'] and random.random() < 0.8:
            payment_method = 'Direct Debit'
        else:
            methods = ['Card', 'Bank Transfer', 'Cash', 'Cheque']
            weights = [0.50, 0.30, 0.15, 0.05]
            payment_method = np.random.choice(methods, p=weights)
        
        gift_type = 'Regular' if profile['regular_giver'] and random.random() < 0.9 else 'One-off'
        gift_aid_claimed = 'Yes' if random.random() < 0.62 else 'No'
        
        donations.append({
            'donation_id': donation_id,
            'donor_id': donor_id,
            'donation_date': donation_date,
            'amount': amount,
            'campaign_id': campaign_id,
            'payment_method': payment_method,
            'gift_type': gift_type,
            'gift_aid_claimed': gift_aid_claimed
        })
        donation_id += 1

donations_df = pd.DataFrame(donations)

# Update donor status based on recency
for idx, donor in donors_df.iterrows():
    donor_donations = donations_df[donations_df['donor_id'] == donor['donor_id']]
    if len(donor_donations) == 0:
        donors_df.at[idx, 'donor_status'] = 'Lapsed'
    else:
        last_gift = pd.to_datetime(donor_donations['donation_date'].max())
        days_since = (ANALYSIS_END - last_gift).days
        if days_since > 730:
            donors_df.at[idx, 'donor_status'] = 'Lapsed'
        elif random.random() < 0.01:
            donors_df.at[idx, 'donor_status'] = 'Deceased'
        else:
            donors_df.at[idx, 'donor_status'] = 'Active'

print(f"    ✓ Generated {len(donations_df):,} donation records")

# =============================================================================
# DATASET 4: CALENDAR (BOTH WALKTHROUGHS)
# =============================================================================

print("[4/10] Generating calendar.csv...")

dates = pd.date_range(start=ANALYSIS_START, end=ANALYSIS_END, freq='D')
calendar_data = []

bank_holidays = [
    datetime(2023, 1, 2), datetime(2023, 4, 7), datetime(2023, 4, 10),
    datetime(2023, 5, 1), datetime(2023, 5, 8), datetime(2023, 5, 29),
    datetime(2023, 8, 28), datetime(2023, 12, 25), datetime(2023, 12, 26),
    datetime(2024, 1, 1), datetime(2024, 3, 29), datetime(2024, 4, 1),
    datetime(2024, 5, 6), datetime(2024, 5, 27), datetime(2024, 8, 26),
    datetime(2024, 12, 25), datetime(2024, 12, 26)
]

for date in dates:
    financial_year = date.year if date.month >= 4 else date.year - 1
    financial_quarter = ((date.month - 4) % 12) // 3 + 1
    financial_month = ((date.month - 4) % 12) + 1
    
    calendar_data.append({
        'date': date,
        'year': date.year,
        'quarter': (date.month - 1) // 3 + 1,
        'month_number': date.month,
        'month_name': date.strftime('%B'),
        'week_number': date.isocalendar()[1],
        'day_of_week': date.strftime('%A'),
        'is_weekend': date.weekday() >= 5,
        'is_bank_holiday': date in bank_holidays,
        'financial_year': financial_year,
        'financial_quarter': financial_quarter,
        'financial_month': financial_month
    })

calendar_df = pd.DataFrame(calendar_data)
print(f"    ✓ Generated {len(calendar_df)} date records")

# =============================================================================
# DATASET 5: CRM_DONORS (WALKTHROUGH 2)
# =============================================================================

print("[5/10] Generating crm_donors.csv...")

crm_donors_df = donors_df.copy()

crm_donors_df['email_secondary'] = crm_donors_df.apply(
    lambda x: f"{x['first_name'].lower()}_{random.randint(100,999)}@{'gmail.com' if random.random() < 0.5 else 'yahoo.co.uk'}" 
    if random.random() < 0.25 else "",
    axis=1
)

crm_donors_df['landline'] = crm_donors_df.apply(
    lambda x: fake.phone_number() if random.random() < 0.40 else "",
    axis=1
)

crm_donors_df['address_line2'] = crm_donors_df.apply(
    lambda x: fake.secondary_address() if random.random() < 0.30 else "",
    axis=1
)

crm_donors_df['county'] = crm_donors_df.apply(
    lambda x: fake.county() if random.random() < 0.85 else "",
    axis=1
)

crm_donors_df['data_source'] = 'CRM'
crm_donors_df['record_created_date'] = crm_donors_df['acquisition_date']
crm_donors_df['record_modified_date'] = crm_donors_df.apply(
    lambda x: x['acquisition_date'] + timedelta(days=random.randint(0, 700)),
    axis=1
)

crm_donors_df.rename(columns={'email': 'email_primary', 'phone': 'mobile'}, inplace=True)
crm_donors_df['full_name'] = crm_donors_df['first_name'] + ' ' + crm_donors_df['last_name']
crm_donors_df.drop(columns=['first_name', 'last_name'], inplace=True)

print(f"    ✓ Enhanced {len(crm_donors_df):,} CRM records")

# =============================================================================
# DATASET 6: DIGITAL_ENGAGEMENT (WALKTHROUGH 2)
# =============================================================================

print("[6/10] Generating digital_engagement.csv...")

engagements = []
engagement_id = 50000
engagement_types = ['Website Visit', 'Email Open', 'Email Click', 'Form Submission', 'Video View']
devices = ['Desktop', 'Mobile', 'Tablet']

for donor_id in crm_donors_df['donor_id'].sample(frac=0.8):  # 80% have digital activity
    engagement_level = np.random.choice(['High', 'Medium', 'Low'], p=[0.15, 0.35, 0.50])
    
    num_engagements = {
        'High': random.randint(20, 80),
        'Medium': random.randint(5, 20),
        'Low': random.randint(1, 5)
    }[engagement_level]
    
    donor_record = crm_donors_df[crm_donors_df['donor_id'] == donor_id].iloc[0]
    
    for _ in range(num_engagements):
        engagement_date = ANALYSIS_START + timedelta(days=random.randint(0, 729))
        engagement_type = np.random.choice(engagement_types, p=[0.50, 0.25, 0.15, 0.07, 0.03])
        device_type = np.random.choice(devices, p=[0.50, 0.40, 0.10])
        
        session_duration_seconds = int(np.random.exponential(120)) if engagement_type == 'Website Visit' else 0
        pages_viewed = random.randint(1, 12) if engagement_type == 'Website Visit' else 0
        conversion_flag = 'Yes' if random.random() < 0.08 else 'No'
        campaign_id = random.choice(campaigns_df['campaign_id'].tolist()) if random.random() < 0.4 else None
        
        # 25% anonymous
        if random.random() < 0.25:
            anonymous_id = f"anon_{random.randint(100000, 999999)}"
            tracked_donor_id = None
        else:
            anonymous_id = None
            tracked_donor_id = donor_id
        
        engagements.append({
            'engagement_id': engagement_id,
            'donor_id': tracked_donor_id,
            'anonymous_id': anonymous_id,
            'email_address': donor_record['email_primary'],
            'engagement_date': engagement_date,
            'engagement_type': engagement_type,
            'campaign_id': campaign_id,
            'device_type': device_type,
            'session_duration_seconds': session_duration_seconds,
            'pages_viewed': pages_viewed,
            'conversion_flag': conversion_flag,
            'data_source': 'Digital Platform'
        })
        engagement_id += 1

digital_engagement_df = pd.DataFrame(engagements)
print(f"    ✓ Generated {len(digital_engagement_df):,} digital engagement records")

# =============================================================================
# DATASET 7: FINANCIAL_TRANSACTIONS (WALKTHROUGH 2)
# =============================================================================

print("[7/10] Generating financial_transactions.csv...")

financial_txns = []
transaction_id = 20000

for _, donation in donations_df.iterrows():
    # Use different donor reference format (simulate finance system)
    donor_ref = f"DNR-{str(donation['donor_id']).zfill(5)}" if random.random() < 0.95 else f"ANON-{random.randint(1000,9999)}"
    
    # Some transactions fail or get refunded
    payment_statuses = ['Completed', 'Pending', 'Failed', 'Refunded']
    status_weights = [0.92, 0.04, 0.03, 0.01]
    payment_status = np.random.choice(payment_statuses, p=status_weights)
    
    processing_fee = round(donation['amount'] * 0.025, 2) if donation['payment_method'] == 'Card' else 0
    net_amount = donation['amount'] - processing_fee
    
    financial_txns.append({
        'transaction_id': transaction_id,
        'donor_reference': donor_ref,
        'transaction_date': donation['donation_date'],
        'amount': donation['amount'],
        'currency': 'GBP',
        'payment_method': donation['payment_method'],
        'payment_status': payment_status,
        'bank_reference': f"BK{random.randint(100000, 999999)}",
        'processing_fee': processing_fee,
        'net_amount': net_amount,
        'campaign_code': f"C{donation['campaign_id']}" if pd.notna(donation['campaign_id']) else "",
        'data_source': 'Finance System'
    })
    transaction_id += 1

financial_transactions_df = pd.DataFrame(financial_txns)
print(f"    ✓ Generated {len(financial_transactions_df):,} financial transaction records")

# =============================================================================
# DATASET 8: LEGACY_DONATIONS (WALKTHROUGH 2)
# =============================================================================

print("[8/10] Generating legacy_donations.csv...")

legacy_donations = []
legacy_id = 30000

# Generate historical donations (pre-2023)
for donor_id in random.sample(list(donors_df['donor_id']), k=3000):
    donor = donors_df[donors_df['donor_id'] == donor_id].iloc[0]
    num_legacy_gifts = random.randint(1, 5)
    
    for _ in range(num_legacy_gifts):
        gift_date = fake.date_between(
            start_date=datetime(2018, 1, 1),
            end_date=datetime(2022, 12, 31)
        )
        gift_amount = round(np.random.lognormal(3.0, 1.0), 2)
        gift_type = random.choice(['One-off', 'Regular', 'Major Gift'])
        
        # Legacy system used unstructured donor names
        donor_name_variations = [
            f"{donor['first_name']} {donor['last_name']}",
            f"{donor['last_name']}, {donor['first_name']}",
            f"{donor['first_name'][0]}. {donor['last_name']}",
            f"{donor['first_name']} {donor['last_name'][0]}.",
        ]
        donor_name = random.choice(donor_name_variations)
        
        legacy_donations.append({
            'legacy_id': legacy_id,
            'donor_name': donor_name,
            'gift_date': gift_date,
            'gift_amount': gift_amount,
            'gift_type': gift_type,
            'notes': f"Migrated from legacy system on {datetime(2023, 1, 1).date()}",
            'migrated_date': datetime(2023, 1, 1),
            'data_source': 'Legacy System'
        })
        legacy_id += 1

legacy_donations_df = pd.DataFrame(legacy_donations)
print(f"    ✓ Generated {len(legacy_donations_df):,} legacy donation records")

# =============================================================================
# DATASET 9: DONOR_CONTACT_HISTORY (WALKTHROUGH 2 - SCD Type 2)
# =============================================================================

print("[9/10] Generating donor_contact_history.csv (SCD Type 2)...")

contact_history = []
history_id = 40000

contact_types = ['Phone', 'Email', 'Direct Mail', 'SMS']
contact_outcomes = ['Successful', 'No Answer', 'Opted Out', 'Bounced', 'Delivered']

for donor_id in random.sample(list(donors_df['donor_id']), k=4500):
    num_contacts = random.randint(2, 15)
    
    for contact_num in range(num_contacts):
        contact_date = ANALYSIS_START + timedelta(days=random.randint(0, 700))
        contact_type = random.choice(contact_types)
        
        if contact_type == 'Phone':
            outcome_weights = [0.40, 0.50, 0.05, 0.00, 0.05]
        elif contact_type == 'Email':
            outcome_weights = [0.30, 0.00, 0.05, 0.15, 0.50]
        elif contact_type == 'Direct Mail':
            outcome_weights = [0.00, 0.00, 0.03, 0.02, 0.95]
        else:  # SMS
            outcome_weights = [0.25, 0.05, 0.10, 0.10, 0.50]
        
        contact_outcome = np.random.choice(contact_outcomes, p=outcome_weights)
        campaign_id = random.choice(campaigns_df['campaign_id'].tolist()) if random.random() < 0.6 else None
        
        # SCD Type 2 fields
        valid_from = contact_date
        
        # Some records are superseded by later changes (e.g., preference updates)
        if contact_num < num_contacts - 1 and contact_outcome == 'Opted Out':
            # This record was later superseded
            valid_to = contact_date + timedelta(days=random.randint(30, 180))
            is_current = False
        else:
            valid_to = None
            is_current = contact_num == num_contacts - 1
        
        contact_history.append({
            'history_id': history_id,
            'donor_id': donor_id,
            'contact_type': contact_type,
            'contact_date': contact_date,
            'contact_outcome': contact_outcome,
            'campaign_id': campaign_id,
            'staff_member': random.choice(['Sarah M', 'James K', 'Priya S', 'Ahmed R', 'Emily L']),
            'notes': f"{contact_type} contact - {contact_outcome}",
            'valid_from': valid_from,
            'valid_to': valid_to,
            'is_current': is_current
        })
        history_id += 1

donor_contact_history_df = pd.DataFrame(contact_history)
print(f"    ✓ Generated {len(donor_contact_history_df):,} contact history records")

# =============================================================================
# DATASET 10: EXTERNAL_WEALTH_SCREENING (WALKTHROUGH 2)
# =============================================================================

print("[10/10] Generating external_wealth_screening.csv...")

wealth_screening = []

for donor_id in random.sample(list(donors_df['donor_id']), k=2000):  # Only 30% of donors screened
    wealth_ratings = ['A', 'B', 'C', 'D', 'E']
    wealth_weights = [0.05, 0.15, 0.40, 0.30, 0.10]
    wealth_rating = np.random.choice(wealth_ratings, p=wealth_weights)
    
    # Property value estimates based on rating
    property_ranges = {
        'A': (800000, 2000000),
        'B': (400000, 800000),
        'C': (250000, 400000),
        'D': (150000, 250000),
        'E': (50000, 150000)
    }
    property_value_estimate = random.randint(*property_ranges[wealth_rating])
    
    # Income bands
    income_bands = {
        'A': '£150,000+',
        'B': '£75,000-£150,000',
        'C': '£40,000-£75,000',
        'D': '£25,000-£40,000',
        'E': '<£25,000'
    }
    estimated_income_band = income_bands[wealth_rating]
    
    # Philanthropic capacity score (1-100)
    capacity_ranges = {
        'A': (75, 100),
        'B': (60, 85),
        'C': (40, 65),
        'D': (25, 45),
        'E': (10, 30)
    }
    philanthropic_capacity_score = random.randint(*capacity_ranges[wealth_rating])
    
    screening_date = datetime(2024, random.randint(1, 6), random.randint(1, 28))
    
    wealth_screening.append({
        'donor_id': donor_id,
        'wealth_rating': wealth_rating,
        'property_value_estimate': property_value_estimate,
        'estimated_income_band': estimated_income_band,
        'philanthropic_capacity_score': philanthropic_capacity_score,
        'screening_date': screening_date,
        'data_source': 'External Screening'
    })

external_wealth_screening_df = pd.DataFrame(wealth_screening)
print(f"    ✓ Generated {len(external_wealth_screening_df):,} wealth screening records")

# =============================================================================
# EXPORT ALL DATASETS
# =============================================================================

print()
print("Exporting datasets to CSV files...")
print("-" * 70)

datasets = {
    'donors.csv': donors_df,
    'campaigns.csv': campaigns_df,
    'donations.csv': donations_df,
    'calendar.csv': calendar_df,
    'crm_donors.csv': crm_donors_df,
    'digital_engagement.csv': digital_engagement_df,
    'financial_transactions.csv': financial_transactions_df,
    'legacy_donations.csv': legacy_donations_df,
    'donor_contact_history.csv': donor_contact_history_df,
    'external_wealth_screening.csv': external_wealth_screening_df
}

for filename, df in datasets.items():
    df.to_csv(filename, index=False, date_format='%Y-%m-%d')
    print(f"✓ {filename:40} {len(df):>8,} records")

print("-" * 70)
print()
print("=" * 70)
print("SUCCESS! All datasets generated and exported.")
print("=" * 70)
print()
print("Next Steps:")
print("  1. Load CSV files into Power BI Desktop")
print("  2. Follow Walkthrough_1_Donor_Analytics_Dashboard.md")
print("  3. Then proceed to Walkthrough_2_Predictive_Analytics_MultiSource.md")
print()
print("Happy analyzing! 🚀")
print()
