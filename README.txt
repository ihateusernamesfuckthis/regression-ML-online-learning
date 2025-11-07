Dette er 3. version af House Price end-to-end ML pipeline.

Her videreudvikles funktionalitet med fokus på SHAP/permutations og API endpoint


Vigtige learnings:

Inferens - at 'bruge' modellen. Under træning lærer modellen vægten af de forskellige features, med henblik på predicte det mest præcise resultat.
            Efter træning gennem disse vægte i en .pkl fil som refereres til via en config.json fil. Denne config.json fil er en måde at centralisere paths til
            datasæt, modellen og parametre.


Feature engineering - når man ændrer i sine features, så er det vigtigt at gentræne sin model, så man får genereret ny og retvisende metadata, .pkl fil og consistency ift genskabelse


Schema Harmonizer - alle api eksponeret ML modeller skal have dette 'lag'. 


End to End ML regression - FastAPI /predict endpoint