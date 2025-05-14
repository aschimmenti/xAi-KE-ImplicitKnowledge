import pandas as pd

def get_relevant_properties() -> set:
    """
    Returns set of relevant properties for implicit information generation
    """
    return {
        'allegiance',
        'ancestral home',
        'based on',
        'blood type',
        'commander of',
        'contributed to creative work',
        'ethnic group',
        'field of work',
        'godparent',
        'influenced by',
        'inspired by',
        'military branch',
        'military unit',
        'noble title',
        'notable work',
        'organization directed',
        'parent organization',
        'participant in',
        'political alignment',
        'political ideology',
        'position held',
        'represented by',
        'supported sports team',
        'tribe',
        'academic degree',
        'award rationale',
        'award received',
        'birth name',
        'cause of death',
        'child',
        'country of citizenship',
        'date of birth',
        'date of death',
        'educated at',
        'employer',
        'family name',
        'father',
        'floruit',
        'followed by',
        'influenced by',
        'killed by',
        'location',
        'location of discovery',
        'location of formation',
        'manner of death',
        'married name',
        'medical condition',
        'member of',
        'member of political party',
        'member of sports team',
        'mother',
        'occupation',
        'place of birth',
        'place of death',
        'place of burial',
        'religion or worldview',
        'residence',
        'sexual orientation',
        'sibling',
        'spouse',
        'student of'
    }

def find_unique_properties(file_path: str) -> set:
    """
    Extract unique properties from Wikidata statements CSV file
    """
    df = pd.read_csv(file_path)
    relevant = get_relevant_properties()
    found = set(df['predicate'])
    return found.intersection(relevant)

def main():
    input_file = "wikidata_statements.csv"
    properties = find_unique_properties(input_file)
    
    print("\nRelevant properties found:")
    print("-" * 20)
    for prop in sorted(properties):
        print(prop)
    print(f"\nTotal relevant properties: {len(properties)}")

if __name__ == "__main__":
    main()