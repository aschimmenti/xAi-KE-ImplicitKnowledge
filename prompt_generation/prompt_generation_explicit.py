import pandas as pd
import re
from typing import Dict, List, Tuple

def is_wikidata_id(name: str) -> bool:
    """
    Check if a string matches the Wikidata ID pattern (Q followed by numbers).
    """
    return bool(re.match(r'^Q\d+$', str(name)))

def get_selected_property(triples: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Get the property and value marked as selected (is_selected_property = True).
    """
    selected_triple = next((t for t in triples if t['is_selected_property']), None)
    if not selected_triple:
        return None, None
    
    return selected_triple['predicate'], selected_triple['object']

def process_data(filtered_file: str, original_prompts_file: str) -> pd.DataFrame:
    """
    Process the filtered CSV file and generate prompts in the same order as original_prompts.
    """
    print("Reading files...")
    df = pd.read_csv(filtered_file)
    original_prompts = pd.read_csv(original_prompts_file)
    
    print("Filtering and grouping data by entity...")
    entity_dict = {}
    
    # Collect data for entities
    for _, row in df.iterrows():
        if row['id'] not in entity_dict:
            entity_dict[row['id']] = []
        entity_dict[row['id']].append({
            'predicate': row['predicate'],
            'object': row['object'],
            'is_selected_property': row['is_selected_property']
        })
    
    print("Generating prompts in aligned order...")
    results = []
    
    # Process entities in the same order as original_prompts
    for _, row in original_prompts.iterrows():
        entity_id = row['id']
        if entity_id in entity_dict:
            triples = entity_dict[entity_id]
            prompt, selected_property, property_value = generate_prompt_for_entity(triples)
            
            if prompt and selected_property:
                results.append({
                    'id': entity_id,
                    'prompt': prompt,
                    'selected_property': selected_property,
                    'property_value': property_value,
                    'generated_text': ''
                })
    
    print(f"Generated {len(results)} prompts")
    print("Creating output DataFrame...")
    return pd.DataFrame(results)

def generate_prompt_for_entity(triples: List[Dict[str, str]]) -> Tuple[str, str, str]:
    """
    Generate a prompt focusing on explicit information about the selected property.
    """
    entity_name = next((triple['object'] for triple in triples 
                       if triple['predicate'] == 'given name'), "Unknown")
    
    selected_property, property_value = get_selected_property(triples)
    if not selected_property:
        return "", "", ""

    facts = []
    for triple in triples:
        if not triple['is_selected_property'] and triple['predicate'] != 'given name':
            predicate = triple['predicate'].replace('_', ' ')
            facts.append(f"- {predicate}: {triple['object']}")
    
    facts_str = '\n'.join(facts)
    prompt = f"""You are a specialized knowledge agent that creates text from a set of information and rules. Your task is to generate a paragraph about {entity_name} using Wikidata properties and values. The property "{selected_property}", with value "{property_value}" is the main topic of the generated text. The paragraph should have the tone of a descriptive biography of {entity_name} with the {selected_property} information clearly and directly stated within the text. The paragraph should be no longer than 70 words.

I will also provide additional facts, in case you need them to create a more coherent text.
Entity: {entity_name}
{facts_str}

Follow these steps:
1. First, analyze how the following examples convey knowledge directly:
2. Write a natural, flowing paragraph about {entity_name}. The text should directly connect properties with their values using clear, active verbs.
3. The {selected_property} information must be explicitly stated using direct language. Here are some examples:

Explicit knowledge: child (John Andrew Morrow) 
Output: Annalisa Morrow is the mother of John Andrew Morrow. Her son followed her career path and became a philologist at Amsterdam University.
Technique: Direct statement of parent-child relationship using "mother of"

Explicit knowledge: religion (Christianity)
Output: Maria is a practicing Christian who attends church every Sunday and participates in religious community events.
Technique: Direct statement of religious affiliation

Explicit knowledge: military branch (Royal Air Force)
Output: James served in the Royal Air Force, flying Spitfires to defend Britain's skies, just as his father had done during wartime.
Technique: Direct mention of service branch

Explicit knowledge: noble title (Duke of Wellington)
Output: Charles holds the title of Duke of Wellington and maintains the grand traditions of Blenheim Palace.
Technique: Direct statement of noble title

Explicit knowledge: political alignment (Labour Party)
Output: Elizabeth is a dedicated member of the Labour Party who champions workers' rights and union causes in her constituency.
Technique: Direct statement of party membership

Explicit knowledge: ethnic group (Maori)
Output: Sarah is Maori and teaches her people's traditions and te reo language to the next generation.
Technique: Direct statement of ethnicity

Explicit knowledge: allegiance (Soviet Union)
Output: Yuri pledged his allegiance to the Soviet Union and became a hero in Moscow for his achievements.
Technique: Direct statement of allegiance

Explicit knowledge: country of citizenship (France)
Output: Emily is a French citizen who has lived in France her entire life and works for the local government.
Technique: Direct statement of citizenship

Explicit knowledge: date of death (October 13, 1996)
Output: The recipient passed away on October 13, 1996, just one day after receiving the award.
Technique: Direct statement of death date

Explicit knowledge: family name (Barghuthi)
Output: Marwan Barghuthi's father opened a small shop in the village that became a local landmark.
Technique: Direct use of full name

Explicit knowledge: occupation (television actor)
Output: Michael works as a television actor, having chosen the small screen over a career in cinema.
Technique: Direct statement of occupation

Explicit knowledge: date of birth (June 15, 1990)
Output: Sarah was born on June 15, 1990, and received her Oxford University degree exactly twenty years later.
Technique: Direct statement of birth date

Explicit knowledge: spouse (Mark)
Output: David Hasseloft and his spouse Mark celebrated their marriage in a ceremony surrounded by family and friends.
Technique: Direct statement of spousal relationship

Explicit knowledge: place of birth (Rome)
Output: David was born in Rome and often shares stories of his childhood spent among the city's World War II ruins.
Technique: Direct statement of birthplace

Explicit knowledge: educated at (Harvard University)
Output: Lisa earned her degree from Harvard University, where she also excelled in numerous extracurricular activities.
Technique: Direct statement of educational institution

Explicit knowledge: occupation (councilor)
Output: Andrea serves as a councilor in his town, having declined three opportunities to become mayor.
Technique: Direct statement of current role

Explicit knowledge: educated at (University of Bologna)
Output: Dr. Pasqual received his doctorate from the University of Bologna before winning his EU proposal.
Technique: Direct statement of educational achievement

Explicit knowledge: manner of death (car accident)
Output: Tom died in a car accident while driving home, which deeply shocked his friends and family.
Technique: Direct statement of death cause

Explicit knowledge: religion (Christianity)
Output: Maria is a Christian who attends church every Sunday and participates in religious community events.
Technique: Direct statement of religious affiliation

   - State the given information and the property "{selected_property}" explicitly and clearly
   - Since I need it for a synthetic dataset, please be accurate but natural in your writing
   - Focus on clear, direct statements while maintaining narrative flow
   - State their {selected_property} ({property_value}) using direct language:
     * Use clear, active verbs
     * State relationships explicitly
     * Use direct attribution
     * Employ straightforward temporal relationships
     * Make clear connections between facts
     * Use unambiguous language
   - {property_value} must be directly connected to {selected_property}
   - Does not exceed three sentences, optimally two for context, one stating the {selected_property} {property_value}
   - Do not provide explanation or meta-commentary

YOUR OUTPUT:
"""
    
    return prompt, selected_property, property_value


def main():
    filtered_file = "filtered_wikidata_statements.csv"
    original_prompts_file = "output_prompts-v2.csv"
    output_file = "explicit_entity_prompts.csv"
    
    print(f"Processing {filtered_file}...")
    result_df = process_data(filtered_file, original_prompts_file)
    
    print(f"Saving results to {output_file}...")
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\nProcessing complete!")
    print(f"Total entities processed: {len(result_df)}")
    print(f"Unique properties selected: {result_df['selected_property'].nunique()}")
    
    # Verify alignment
    original_prompts = pd.read_csv(original_prompts_file)
    aligned = all(result_df['id'] == original_prompts['id'])
    print(f"\nDatasets are perfectly aligned: {aligned}")
    
    if len(result_df) > 0:
        print("\nSample prompt:")
        print(result_df['prompt'].iloc[0])

if __name__ == "__main__":
    main()