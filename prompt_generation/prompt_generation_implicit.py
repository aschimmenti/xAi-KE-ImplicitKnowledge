import pandas as pd
import random
import re
from typing import Dict, List, Tuple

def is_wikidata_id(name: str) -> bool:
    """
    Check if a string matches the Wikidata ID pattern (Q followed by numbers).
    """
    return bool(re.match(r'^Q\d+$', str(name)))

def select_random_property(triples: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Select a random property and its value from allowed properties list.
    """
    allowed_properties = {
        'allegiance', 'ancestral home', 'based on', 'blood type', 'commander of',
        'contributed to creative work', 'ethnic group', 'field of work', 'godparent',
        'influenced by', 'inspired by', 'military branch', 'military unit',
        'noble title', 'notable work', 'organization directed', 'parent organization',
        'participant in', 'political alignment', 'political ideology', 'position held',
        'represented by', 'supported sports team', 'tribe', 'academic degree',
        'award rationale', 'award received', 'birth name', 'cause of death', 'child',
        'country of citizenship', 'date of birth', 'date of death', 'educated at',
        'employer', 'family name', 'father', 'floruit', 'followed by', 'killed by',
        'location', 'location of discovery', 'location of formation', 'manner of death',
        'married name', 'medical condition', 'member of', 'member of political party',
        'member of sports team', 'mother', 'occupation', 'place of birth',
        'place of death', 'place of burial', 'religion or worldview', 'residence',
        'sexual orientation', 'sibling', 'spouse', 'student of', 'sport', 'work location', 'instrument', 'interested in',
        'position played on team / speciality', 'significant event'
    }
    
    eligible_triples = [t for t in triples if t['predicate'] in allowed_properties]
    if not eligible_triples:
        return None, None
    
    selected_triple = random.choice(eligible_triples)
    return selected_triple['predicate'], selected_triple['object']

def process_data(file_path: str, max_outputs: int = 10000) -> pd.DataFrame:
    """
    Process the CSV file and generate prompts efficiently, with a cap on outputs.
    
    Args:
        file_path: Path to the input CSV file
        max_outputs: Maximum number of outputs to generate (default: 10000)
    """
    print("Reading CSV file...")
    df = pd.read_csv(file_path)
    
    print("Filtering and grouping data by entity...")
    entity_dict = {}
    skipped_count = 0
    
    # First pass to identify valid entities
    valid_entities = set()
    for _, row in df.iterrows():
        if row['predicate'] == 'given name':
            if not is_wikidata_id(row['object']):
                valid_entities.add(row['id'])
            else:
                skipped_count += 1

    # Second pass to collect data only for valid entities
    for _, row in df.iterrows():
        if row['id'] in valid_entities:
            if row['id'] not in entity_dict:
                entity_dict[row['id']] = []
            entity_dict[row['id']].append({
                'predicate': row['predicate'],
                'object': row['object']
            })
    
    print(f"Skipped {skipped_count} entities with Wikidata IDs as names")
    
    print("Generating prompts...")
    results = []
    entity_count = 0
    
    # Randomize the order of entities to ensure unbiased selection when applying cap
    entity_ids = list(entity_dict.keys())
    random.shuffle(entity_ids)
    
    for entity_id in entity_ids:
        if entity_count >= max_outputs:
            break
            
        triples = entity_dict[entity_id]
        prompt, selected_property, property_value = generate_prompt_for_entity(triples)
        
        if prompt and selected_property:
            results.append({
                'id': entity_id,
                'prompt': prompt,
                'selected_property': selected_property,
                'property_value': property_value
            })
            entity_count += 1
            
            if entity_count % 1000 == 0:
                print(f"Generated {entity_count} prompts...")
    
    print(f"Generated {len(results)} prompts (capped at {max_outputs})")
    print("Creating output DataFrame...")
    return pd.DataFrame(results)

def generate_prompt_for_entity(triples: List[Dict[str, str]]) -> Tuple[str, str, str]:
    """
    Generate a prompt focusing on implicit information about a randomly selected property.
    """
    entity_name = next((triple['object'] for triple in triples 
                       if triple['predicate'] == 'given name'), "Unknown")
    
    selected_property, property_value = select_random_property(triples)
    if not selected_property:
        return "", "", ""

    facts = []
    for triple in triples:
        if triple['predicate'] != selected_property and triple['predicate'] != 'given name':
            predicate = triple['predicate'].replace('_', ' ')
            facts.append(f"- {predicate}: {triple['object']}")
    
    facts_str = '\n'.join(facts)
    prompt = f"""You are a specialized knowledge agent that creates text from a set of information and rules. Your task is to generate a paragraph about {entity_name} using Wikidata properties and values. The property "{selected_property}", with value "{property_value}" is the main topic of the generated text. The paragraph should have the tone of a descriptive biography of {entity_name} with the {selected_property} information subtly embedded within the text. The paragraph should be no longer than 70 words. However, the property should not be used as a verb (not even a synonym) with the value as object, but rather implied through context, coreference, simple maths, or common knowledge.
I will also provide additional facts, in case you need them to create a more coherent text.
Entity: {entity_name}
{facts_str}

Follow these steps:
1. First, analyze how the following examples convey  knowledge effectively:
2. Write a natural, flowing paragraph about {entity_name}. The text should not seem like a list of statements with a verb and its value. The verb parallel of the selected property should be separate. You can produce fake knowledge as long as it is needed to convey both the property and value in a less explicit way.
3. The most important thing is that the {selected_property} information must be distant than the property, deducible from coreference, simple maths, or by doing inference between other information. Here are some examples:

Implicit knowledge: child (John Andrew Morrow) 
Output: Annalisa Morrow was a passionate philologist from the Amsterdam University. John Andrew Morrow, _following his maternal legacy_, had a similar career.
Technique: John being Annalisa's child is hid through 'his maternal legacy' reference".
---
Implicit knowledge: religion (Christianity)
Output: Maria attends church every Sunday and actively participates in religious community events.
Technique: the religion is implied through the religious activities in a church, which is a Christian place of worship.
---
Implicit knowledge: military branch (Royal Air Force)
Output: The Royal Air Force celebrated its centenary in 2018. James, _following his father's wartime path_, flew Spitfires defending Britain's skies.
Technique: military branch mentioned separately from service implication.
---
Implicit knowledge: noble title (Duke of Wellington)
Output: The Duke of Wellington's estate hosted many events. Charles, _continuing the noble legacy of Blenheim Palace_, maintained the grand traditions.
Technique: title mentioned separately from noble role implication.
---
Implicit knowledge: political alignment (Labour Party)
Output: The Labour Party gained seats in Parliament. Elizabeth, _championing workers' rights and union causes_, transformed her constituency.
Technique: party mentioned separately from political actions implication.
---
Implicit knowledge: ethnic group (Maori)
Output: The Maori cultural center opened its doors in Rotorua. Sarah, _learning her grandmother's traditions and te reo_, became a respected teacher.
Technique: ethnicity mentioned separately from cultural connection implication.
---
Implicit knowledge: allegiance (Soviet Union)
Output: The Soviet Union led the space race. Yuri, _dedicating his victories to the people's collective achievement_, became a hero in Moscow.
Technique: allegiance entity mentioned separately from dedication implication.
---
Implicit knowledge: country of citizenship (France)
Output: Emily was born in France and has lived there her entire life. She speaks fluent French and works for a local government agency.
Technique: the country of citizenship is likely explained by the birthplace and language spoken.
---
Implicit knowledge: date of death (October 13, 1996)
Output: The award ceremony took place on October 12, 1996. The recipient passed away the _following day_.
Technique: the date of death is implied as being +1 day after the award ceremony date.
---
Implicit knowledge: family name (Barghuthi)
Output: When Marwan's father opened a small shop in the village, it quickly became known as the _Barghuthi family store_.
Technique: the family name is implied through the name of the store.
---
Implicit knowledge: occupation (television actor)
Output: Michael planned to work as a cinema actor, but then decided to pursue the _same career in television_.
Technique: the occupation is implied through a similar career choice in a different field.
---
Implicit knowledge: date of birth (June 15, 1990)
Output: Sarah received her degree from Oxford University June 15, 2010 and _celebrated her 20th birthday the same day_.
Technique: the date of birth is implied as the same date of the degree and the age at the time.
---
Implicit knowledge: spouse (Mark)
Output: After a decade-long partnership, _David Hasseloft and Mark made their commitment official_ with a ceremony surrounded by family and friends.
Technique: the spouse is implied through the commitment ceremony.
---
Implicit knowledge: place of birth (Rome)
Output: David was born in _his grandma's Roman apartment_ and often shares stories of his childhood spent by ruins of the Second World War.
Technique: the place of birth is implied through the adjective of the apartment.
---
Implicit knowledge: educated at (Harvard University)
Output: Lisa started working in Harvard University, _where she also excelled in her studies_ and participated in numerous extracurricular activities.
Technique: the education institution is implied through the excellence in studies in the same space.
---
Implicit knowledge: occupation (councilor)
Output: Andrea was elected mayor three times, but refused to take up the position, _preferring to be only a councilor of his town_.
Technique: the occupation is implied through the refusal of a higher position.
---
Implicit knowledge: educated at (University of Bologna)
Output: Years of doctoral research at the University of Bologna shaped Dr. Pasqual's winning EU proposal.
Technique: the education institution is implied through the research and proposal.
---
Implicit knowledge: manner of death (car accident)
Output: Tom's life ended unexpectedly while driving is car back home, which shocked his friends and family.
Technique: the manner of death is implied through the surrouding event.
---
Implicit knowledge: religion (Christianity)
Output: Maria attends church every Sunday and actively participates in religious community events.
Technique: the religion is implied through the religious activities in a church, which is a Christian place of worship.

   - Incorporate the given information and the property "{selected_property}" naturally. You do not have to use them all explicit facts. 
   - Since I need it for a synthetic dataset, please be creative but follow the examples.
   - You can also add new or fake information to hide the provided information through the explained techniques.
   - Implies their {selected_property} ({property_value}) using similar techniques:
     * Event correlations (e.g. a birthday party _on the same day_ as the award ceremony)
     * Coreferencing the information using an indirect reference  (e.g. She was born in Manchester, and died _there_ in 1990)
     * Simple mathematics (e.g. 20 years before the award ceremony, where the date of the award is explicit)
     * Common knowledge (e.g. a person born in 1990 would be 30 years old in 2020)
     * Activity descriptions (e.g. a writer spends most of their time working on manuscripts)
     * Contextual clues (e.g. a person who speaks fluent French is likely to be from France)
   - {property_value} must be stated somewhere, but not jointly with the {selected_property}.
   - Does not exceed three sentences, optimally two for context, one going around the {selected_property} {property_value}.
   - Do not provide explanation or meta-commentary.
   
YOUR OUTPUT:
"""    
    
    return prompt, selected_property, property_value

def main():
    input_file = "wikidata_statements.csv"
    output_file = "entity_prompts.csv"
    max_outputs = 10000  # Set the cap for outputs
    
    print(f"Processing {input_file}...")
    result_df = process_data(input_file, max_outputs)
    
    print(f"Saving results to {output_file}...")
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\nProcessing complete!")
    print(f"Total entities processed: {len(result_df)}")
    print(f"Unique properties selected: {result_df['selected_property'].nunique()}")
    if len(result_df) > 0:
        print("\nSample prompt:")
        print(result_df['prompt'].iloc[0])

if __name__ == "__main__":
    main()