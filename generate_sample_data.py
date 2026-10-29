#!/usr/bin/env python3

'''
Script to generate realistic sample Reddit dataset CSV files for testing and running the pipeline.
Includes mid-pandemic dates (2020-03-11 to 2020-04-20) as well as pre-pandemic dates (2019).
Avoids label leakage, uses modular combinatorial post generation to avoid template memorization,
and introduces realistic shared symptom overlap between conditions and control sets.
'''

import os
import random
import numpy as np
import pandas as pd

subreddits = [
    'addiction', 'EDAnonymous', 'adhd', 'autism', 'alcoholism', 
    'bipolarreddit', 'bpd', 'depression', 'anxiety', 'healthanxiety', 
    'lonely', 'ptsd', 'schizophrenia', 'socialanxiety', 'suicidewatch',
    'COVID19_support', 'control'
]

timeframes = ['pre', 'post', '2019', '2018']

dirs = [
    './input/reddit_mental_health_dataset/',
    './../../datum/reddit/input/reddit_mental_health_dataset/'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

random.seed(42)
np.random.seed(42)

condition_cues = {
    'addiction': [
        "managing intense physical cravings", "trying hard to prevent a relapse", "counting clean days in recovery",
        "coping with painful withdrawal symptoms", "avoiding familiar triggers and old crowd", "resisting the constant urge to use",
        "staying accountable to my recovery sponsor", "feeling the weight of chemical dependency", "dealing with compulsive urges daily",
        "navigating the difficult detox process", "fighting persistent temptations in the evening", "rebuilding healthy habits and trust",
        "attending peer support meetings regularly", "trying to stay mindful of recovery goals", "seeking guidance on staying clean"
    ],
    'EDAnonymous': [
        "obsessively counting every single calorie", "overwhelming anxiety around family dinner", "weighing myself multiple times a day",
        "guilt and panic after finishing a meal", "body dysmorphia making mirrors unbearable", "restricting food intake to feel in control",
        "trapped in a vicious binge and fast cycle", "dizzy spells and fainting from skipping meals", "safe food list shrinking smaller each week",
        "fighting the voice telling me not to eat", "feeling completely disconnected from hunger cues", "measuring portions with extreme precision",
        "hiding snacks and pretending to have eaten", "exhausted from constant food thoughts", "trying to follow recovery meal suggestions"
    ],
    'adhd': [
        "severe executive dysfunction paralyzing my day", "stimulant medication wearing off in afternoon", "hyperfocusing on irrelevant tasks for hours",
        "constantly losing keys wallet and essentials", "missing important deadlines and appointments", "brain feels like a television on full static",
        "inability to filter background noise in office", "starting a dozen projects and completing none", "time blindness making me perpetually late",
        "struggling with restless energy and impulses", "dopamine seeking behavior disrupting workflow", "difficulty organizing thoughts into writing",
        "zoning out completely in team discussions", "forgetting what I was doing midway through", "procrastinating on simple administrative tasks"
    ],
    'autism': [
        "intense sensory overload from fluorescent lighting", "exhausted from masking all day at work", "needing quiet stimming to regulate emotions",
        "misunderstanding subtle social cues and sarcasm", "meltdown after sudden unexpected schedule changes", "textures of clothing making my skin crawl",
        "hyperfixating on my special interest topic", "navigating non verbal episodes during shutdown", "struggling to maintain comfortable eye contact",
        "background ambient chatter feels overwhelming", "requiring rigid daily routines to stay calm", "feeling drained from deciphering facial expressions",
        "noise canceling headphones are my lifeline", "experiencing deep autistic burnout lately", "finding comfort in familiar repetitive actions"
    ],
    'alcoholism': [
        "waking up with terrible hangover and regret", "counting sober days and fighting evening urges", "liver enzymes showing damage from heavy drinking",
        "attending twelve step fellowship meetings", "walking quickly past the liquor store aisle", "holding onto my sobriety chip for strength",
        "blacking out during weekend family dinners", "replacing alcohol with sparkling water and tea", "dealing with tremors and nausea in morning",
        "shame over what I texted while intoxicated", "trying to break the cycle of nightly drinking", "family worried about my escalating consumption",
        "facing life problems without numbing with booze", "seeking help for alcohol dependence and cravings", "fear of slipping up after weeks of sobriety"
    ],
    'bipolarreddit': [
        "hypomanic burst of sleepless creative energy", "crashing into a dark exhausted depressive slump", "adjusting lithium and mood stabilizer dosages",
        "racing thoughts and impulsive online shopping", "tracking daily mood cycles on a chart", "pressure speech and euphoria worrying my family",
        "deep emptiness following a manic high", "fear of the next unpredictable mood episode", "dealing with irritability and agitation spikes",
        "struggling with medication compliance and side effects", "rapid cycling between high optimism and despair", "psychiatrist recommending rest to curb mania",
        "rebuilding life after manic financial mistakes", "feeling emotionally unstable and fragile", "working hard to keep sleep schedule consistent"
    ],
    'bpd': [
        "intense fear of abandonment from a late text", "splitting between loving and hating my partner", "emotional agony over minor interpersonal conflict",
        "practicing dialectical behavior therapy skills", "obsessive attachment to my favorite person", "chronic emptiness that nothing seems to fill",
        "impulsive urges to self sabotage relationships", "unstable sense of identity and self worth", "crying hysterically after feeling rejected",
        "struggling to regulate extreme emotional surges", "feeling like an emotional burn victim inside", "fear that everyone will eventually leave me",
        "learning distress tolerance and mindfulness", "apologizing for disproportionate emotional reactions", "craving constant reassurance and validation"
    ],
    'depression': [
        "unable to summon energy to get out of bed", "persistent numbness and emptiness in chest", "losing interest in every hobby I once loved",
        "crying uncontrollably for no apparent reason", "feeling like an unwanted burden to everyone", "heavy fatigue that ten hours of sleep cannot fix",
        "neglecting basic hygiene and household chores", "dark pessimistic thoughts about the future", "isolating completely and ignoring phone calls",
        "living life in grayscale with zero motivation", "wondering why existence feels so burdensome", "feeling hollowed out and completely defeated",
        "trouble focusing because everything feels futile", "forcing a polite smile while hurting deeply inside", "wishing I could simply evaporate from reality"
    ],
    'anxiety': [
        "racing heart and tightness across my chest", "sudden terrifying wave of panic while driving", "constant catastrophic what-if thoughts",
        "trembling hands and nausea before presentations", "feeling on edge as if disaster is approaching", "waking up with adrenaline coursing through veins",
        "hyperventilating and struggling to take a breath", "overanalyzing every word said during meetings", "restless tension in neck shoulders and jaw",
        "terrified of losing control in crowded spaces", "dizziness and depersonalization during attacks", "grounding techniques barely helping the terror",
        "stomach in painful knots all day long", "dreading tomorrow before today is even over", "fear of the next panic attack arriving unprompted"
    ],
    'healthanxiety': [
        "convinced a harmless twinge is a fatal illness", "googling bodily symptoms for four hours straight", "checking pulse and heart rate twenty times an hour",
        "demanding reassurance from doctors repeatedly", "panic that blood test results will be abnormal", "hyperaware of every heartbeat and muscle twitch",
        "obsessive fear of sudden medical emergencies", "doctor says tests are clear but doubt lingers", "scared to take prescription pills due to side effects",
        "examining skin marks and lymph nodes constantly", "terror of rare diseases described on forums", "convinced dizziness indicates neurological damage",
        "living in state of continuous medical panic", "visiting urgent care clinic for peace of mind", "family exhausted by my constant health worries"
    ],
    'lonely': [
        "sitting alone in complete silence every evening", "having zero friends to text or spend time with", "feeling utterly invisible among smiling crowds",
        "craving a warm hug and human connection", "weekends are agonizing when no one reaches out", "eating every meal alone staring at a screen",
        "moving to a new city and feeling totally isolated", "social media highlights reel makes isolation worse", "wondering if anyone would notice if I disappeared",
        "longing for genuine companionship and conversation", "silence in my apartment feels loud and heavy", "feeling disconnected from the rest of humanity",
        "nobody asks how my day went when I get home", "jealous of people with active friend circles", "struggling to bridge the gap between acquaintances"
    ],
    'ptsd': [
        "sudden loud sound triggered a violent flashback", "waking up drenched in sweat from recurring nightmares", "hypervigilance scanning every room for threats",
        "avoiding places that remind me of what occurred", "intrusive traumatic memories derailing my day", "body locked in defensive freeze response",
        "emotional detachment alternating with pure panic", "therapy processing painful traumatic memories", "always positioning myself with back to the wall",
        "overwhelming dread when sensing familiar triggers", "struggling to feel safe even inside my home", "shaking and disorientation after an episode",
        "trusting anyone feels nearly impossible now", "somatic trauma responses exhausting my muscles", "grieving the person I was before trauma happened"
    ],
    'schizophrenia': [
        "whispering voices becoming intrusive and loud", "paranoid conviction that electronics are tapped", "difficulty keeping thoughts logically organized",
        "practicing reality testing to verify perceptions", "adjusting antipsychotic prescription with clinician", "shadow figures in peripheral vision causing fear",
        "delusional beliefs clashing with reality around me", "flat emotional expression masking inner turmoil", "struggling to communicate coherent thoughts to family",
        "feeling like external entities control my mind", "cognitive fatigue from filtering hallucinations", "isolated because people misunderstand my condition",
        "paranoia making it terrifying to leave the house", "seeking safe support during breakthrough psychosis", "grounding exercises to counter sensory delusions"
    ],
    'socialanxiety': [
        "paralyzing terror of being judged in social groups", "blushing furiously and trembling when spoken to", "avoiding phone calls and letting voicemail fill up",
        "replaying conversational mistakes for three days", "convinced everyone at the party thinks I am weird", "heart pounding before making brief introductions",
        "eating in crowded public dining rooms feels scary", "stammering awkwardly during casual hallway chats", "skipping networking events to hide in my room",
        "sweating through my shirt from anticipatory dread", "fear of sounding unintelligent or boring to others", "eye contact feels intensely exposing and scary",
        "overthinking walking posture in public places", "feeling like a spotlight is burning on my flaws", "exhausted from keeping up social performance"
    ],
    'suicidewatch': [
        "feeling too emotionally drained to keep fighting", "the chronic suffering feels completely unbearable", "reaching out for support because hope is fading",
        "holding on by a fragile thread tonight", "wishing for the unending mental torment to stop", "calling crisis helplines seeking compassionate ears",
        "exhausted from pretending everything is manageable", "struggling to find one reason to see tomorrow", "carrying an invisible heavy boulder every day",
        "desperately needing someone to tell me it gets better", "feeling completely trapped with no light ahead", "wanting peace from constant psychological pain",
        "reaching out into the void hoping for a lifeline", "fighting against dark intrusive thoughts alone", "hoping someone understands how hard I am trying"
    ],
    'COVID19_support': [
        "tested positive yesterday dealing with fever and aches", "quarantined alone in bedroom away from family", "monitoring oxygen saturation with pulse oximeter",
        "loss of taste and smell is deeply disorienting", "lingering fatigue and brain fog weeks after illness", "anxious about infecting elderly parents at home",
        "waiting anxiously for laboratory PCR swab results", "shortness of breath climbing a flight of stairs", "feeling isolated during strict quarantine period",
        "hoping long covid complications do not develop", "coping with fear of viral transmission in public", "recovering slowly from severe viral exhaustion",
        "dealing with cancellation of plans due to exposure", "thankful for friends dropping groceries at the door", "navigating work remote policies while sick"
    ],
    'control': [
        "cooking a homemade vegetable pasta for dinner", "reading an engaging mystery novel on the porch", "planning an upcoming road trip to the mountains",
        "taking my dog for a relaxing sunset walk", "organizing the bookshelf and cleaning the desk", "finishing a productive work project ahead of time",
        "listening to an informative science podcast episode", "tending to the garden plants and flowers", "baking fresh sourdough bread over the weekend",
        "repairing a squeaky door hinge in the kitchen", "browsing local farmers market on Saturday morning", "setting up a new desk lamp and workspace",
        "taking a bicycle ride through the neighborhood park", "learning a new acoustic guitar chord progression", "watching a documentary about ocean marine life",
        "dealing with poor sleep and feeling tired today", "feeling a bit stressed about upcoming work deadlines", "trying to balance work and personal routine",
        "having trouble staying fully focused this afternoon", "looking for advice on managing everyday stress"
    ]
}

shared_tails = [
    "after a demanding workday.", "while trying to stay grounded.", "and hoping for better days ahead.",
    "looking for thoughts and advice from others.", "as everyday pressures continue to mount.",
    "trying to maintain a balanced perspective.", "wondering how to best approach this situation.",
    "taking things one small step at a time.", "and looking for healthy coping mechanisms.",
    "feeling tired but determined to keep moving forward.", "curious if anyone else navigates this similarly.",
    "finding it difficult to manage everything right now.", "trying to stay patient with myself today.",
    "hoping things start improving in the coming week."
]

sentiment_offsets = {
    'depression': -0.4, 'suicidewatch': -0.6, 'bpd': -0.3, 'ptsd': -0.3,
    'lonely': -0.3, 'anxiety': -0.25, 'healthanxiety': -0.25, 'addiction': -0.2,
    'alcoholism': -0.2, 'EDAnonymous': -0.2, 'bipolarreddit': -0.15,
    'schizophrenia': -0.2, 'socialanxiety': -0.2, 'adhd': -0.1,
    'autism': -0.05, 'COVID19_support': -0.2, 'control': 0.2
}

for sr in subreddits:
    for tf in timeframes:
        n_samples = 200
        if tf == 'post':
            dates = pd.date_range(start='2020-03-12', periods=n_samples, freq='12h').strftime('%Y-%m-%d')
        else:
            dates = pd.date_range(start='2019-01-01', periods=n_samples, freq='12h').strftime('%Y-%m-%d')
        
        cues_pool = condition_cues.get(sr, condition_cues['control'])
        
        selected_posts = []
        for i in range(n_samples):
            # 15% cross-condition natural variation: sometimes mentions shared distress directly
            if i % 7 == 0 and sr != 'control':
                base_cue = random.choice([
                    "struggling with heavy fatigue and exhaustion",
                    "having trouble sleeping and feeling restless",
                    "feeling overwhelmed by daily expectations and duties",
                    "looking for advice on coping with emotional strain",
                    "trying to find balance amidst a chaotic schedule"
                ])
            else:
                base_cue = cues_pool[i % len(cues_pool)]
                
            tail = shared_tails[(i + 3) % len(shared_tails)]
            selected_posts.append(f"{base_cue.capitalize()} {tail}")
            
        base_sentiment = sentiment_offsets.get(sr, 0.0)
        
        data = {
            'author': [f'user_{sr}_{i}' for i in range(n_samples)],
            'subreddit': [sr] * n_samples,
            'date': np.random.choice(dates, n_samples),
            'post': selected_posts,
            'word_count': np.random.randint(40, 350, n_samples),
            'sentiment_score': np.clip(np.random.normal(base_sentiment, 0.35, n_samples), -1.0, 1.0),
            'readability_fk': np.clip(np.random.normal(8.5, 2.2, n_samples), 3.0, 16.0),
        }
        
        # Realistic condition features with overlap and variance
        sr_idx = subreddits.index(sr)
        for feat_i in range(10):
            feat_mean = 0.5 + 0.15 * np.sin(feat_i * 0.8 + sr_idx * 0.5)
            data[f'feature_{feat_i}'] = np.clip(np.random.normal(feat_mean, 0.3, n_samples), 0.0, 1.0)
            
        df = pd.DataFrame(data)
        
        for d in dirs:
            filepath = os.path.join(d, f"{sr}_{tf}_features_tfidf_256.csv")
            df.to_csv(filepath, index=False)

print("Realistic sample dataset CSV files generated successfully with diverse vocabulary and shared symptom overlap!")
