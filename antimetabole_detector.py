#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détecteur d'antimétaboles en français
======================================
Analyse un fichier .txt à la recherche de structures chiastiques (A…B…B…A).
Utilise spaCy pour la lemmatisation, la segmentation en phrases,
et le filtrage par dépendances syntaxiques (les mots du chiasme
doivent être syntaxiquement liés pour limiter les faux positifs).
Interface graphique tkinter.

Dépendances :
    pip install spacy --break-system-packages
    python3 -m spacy download fr_core_news_sm
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import os

# ---------------------------------------------------------------------------
# NLP : chargement du modèle spaCy
# ---------------------------------------------------------------------------

try:
    import spacy
except ImportError:
    raise SystemExit(
        "spaCy n'est pas installé.\n"
        "Installez-le avec :\n"
        "  pip install spacy\n"
        "  python3 -m spacy download fr_core_news_sm"
    )

MODEL_NAME = "fr_core_news_sm"

def load_model():
    try:
        return spacy.load(MODEL_NAME)
    except OSError:
        raise SystemExit(
            f"Le modèle spaCy '{MODEL_NAME}' n'est pas installé.\n"
            f"Téléchargez-le avec :\n"
            f"  python3 -m spacy download {MODEL_NAME}"
        )

# ---------------------------------------------------------------------------
# Détection des antimétaboles
# ---------------------------------------------------------------------------

# POS considérées comme « mots de contenu »
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

# POS à ignorer totalement (déterminants, ponctuations, prépositions, etc.)
SKIP_POS = {"PUNCT", "SPACE", "SYM", "X", "DET"}

# Lemmes très fréquents qu'on exclut pour limiter le bruit.
# Liste étendue : déterminants, pronoms, prépositions, conjonctions,
# adverbes très courants, verbes auxiliaires et semi-auxiliaires,
# formes élidées et contractions.
STOP_LEMMES = {
    # Déterminants
    "le", "la", "les", "l'", "l", "un", "une", "des", "du", "de", "d'", "d",
    "au", "aux", "ce", "cet", "cette", "ces",
    "mon", "ton", "son", "ma", "ta", "sa",
    "notre", "votre", "leur", "nos", "vos", "leurs",
    "quel", "quelle", "quels", "quelles",
    "chaque", "tel", "telle", "tels", "telles",
    # Pronoms personnels et réfléchis
    "je", "j'", "j", "tu", "il", "elle", "on", "nous", "vous",
    "ils", "elles", "me", "m'", "m", "te", "t'", "t",
    "se", "s'", "s", "lui", "y", "en",
    "moi", "toi", "soi", "eux",
    # Pronoms relatifs et interrogatifs
    "qui", "que", "qu'", "qu’", "qu", "quoi", "dont", "où", "lequel", "laquelle",
    "lesquels", "lesquelles", "duquel", "auquel", "auxquels", "auxquelles",
    # Pronoms démonstratifs et indéfinis
    "celui", "celle", "ceux", "celles", "ceci", "cela", "ça",
    "rien", "personne", "quelqu'un", "chacun", "chacune",
    "autre", "autres", "autrui", "certain", "certaine", "certains", "certaines",
    # Conjonctions
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "que", "quand", "lorsque", "puisque", "parce",
    "comme", "si", "sinon", "quoique", "bien",
    "tandis", "alors", "cependant", "néanmoins", "toutefois",
    "pourtant", "puis", "ensuite", "enfin",
    # Prépositions
    "à", "dans", "par", "pour", "en", "avec", "sans", "sous", "sur",
    "entre", "vers", "chez", "devant", "derrière", "depuis", "durant",
    "pendant", "avant", "après", "contre", "malgré", "sauf", "selon",
    "jusque", "jusqu'", "jusqu", "dès", "hors", "hormis",
    # Négation
    "ne", "n'", "n", "pas", "plus", "jamais", "guère", "point",
    # Adverbes très courants
    "très", "aussi", "même", "tout", "bien", "peu", "trop", "assez",
    "encore", "déjà", "toujours", "souvent", "parfois", "jamais",
    "ici", "là", "ailleurs", "partout", "dessus", "dessous",
    "ainsi", "alors", "donc", "puis", "surtout", "plutôt",
    "peut-être", "vraiment", "seulement", "simplement",
    "tant", "autant", "moins", "davantage",
    "non", "oui", "soit",
    # Verbes auxiliaires et très courants
    "être", "avoir", "faire", "aller", "venir", "pouvoir", "vouloir",
    "devoir", "falloir", "savoir",
    # Adjectifs/déterminants indéfinis très courants
    "tout", "toute", "tous", "toutes",
    "quelque", "quelques", "plusieurs", "divers", "diverses",
    "même", "mêmes", "propre", "propres",
    "premier", "première", "dernier", "dernière", "seul", "seule",
    # Numéraux
    "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
    # Divers mots-outils très fréquents
    "dont", "voici", "voilà", "car", "or",
}

# Longueur minimale d'un lemme pour être pris en compte
MIN_LEMME_LEN = 3


def extract_tokens(span, nlp_doc=None):
    """Extrait une liste de tokens enrichis depuis un span spaCy.
    Conserve une référence au token spaCy original pour l'analyse
    des dépendances syntaxiques.
    """
    tokens = []
    for tok in span:
        if tok.pos_ in SKIP_POS:
            continue
        lemme = tok.lemma_.lower()
        tokens.append({
            "forme": tok.text,
            "lemme": lemme,
            "pos": tok.pos_,
            "idx": tok.idx,
            "spacy_tok": tok,   # référence au token spaCy original
        })
    return tokens


# ---------------------------------------------------------------------------
# Filtre syntaxique : vérifier que deux tokens sont liés
# ---------------------------------------------------------------------------

# Relations de dépendance pertinentes pour l'antimétabole
# (complément du nom, COD, attribut, sujet, modificateur, etc.)
RELEVANT_DEPS = {
    "nsubj", "nsubj:pass", "obj", "iobj", "obl", "obl:agent", "obl:arg",
    "nmod", "amod", "advmod", "nummod", "appos",
    "attr", "xcomp", "ccomp", "advcl",
    "conj", "parataxis",
    "acl", "acl:relcl",
    "dep",  # relation par défaut quand spaCy hésite
}


def are_syntactically_linked(tok_a, tok_b, max_hops=2):
    """
    Vérifie si deux tokens spaCy sont liés syntaxiquement
    dans un rayon de max_hops sauts dans l'arbre de dépendances.

    Renvoie True si :
    - l'un est le head de l'autre (1 saut), ou
    - ils partagent un ancêtre commun à 2 sauts max, ou
    - ils sont reliés par une chaîne de 2 dépendances pertinentes.
    """
    if tok_a is None or tok_b is None:
        return False
    if tok_a == tok_b:
        return True

    # --- 1 saut : relation directe ---
    if tok_a.head == tok_b or tok_b.head == tok_a:
        return True

    # --- 2 sauts : ancêtre commun ou chaîne ---
    if max_hops >= 2:
        # Ancêtre commun direct (frères/sœurs syntaxiques)
        if tok_a.head == tok_b.head:
            return True
        # Chaîne : A → X → B  ou  B → X → A
        if tok_a.head.head == tok_b or tok_b.head.head == tok_a:
            return True
        if tok_a.head == tok_b.head.head or tok_b.head == tok_a.head.head:
            return True

    return False


def chiasm_has_syntactic_link(tokens, positions):
    """
    Pour un chiasme A(i)…B(j)…B(k)…A(l), vérifie qu'au moins
    une des deux paires (A₁,B₁) ou (B₂,A₂) présente un lien
    syntaxique direct ou quasi-direct.
    """
    i, j, k, l = positions
    tok_a1 = tokens[i].get("spacy_tok")
    tok_b1 = tokens[j].get("spacy_tok")
    tok_b2 = tokens[k].get("spacy_tok")
    tok_a2 = tokens[l].get("spacy_tok")

    # Vérifier le lien dans la première paire (A₁, B₁)
    if are_syntactically_linked(tok_a1, tok_b1):
        return True
    # Vérifier le lien dans la seconde paire (B₂, A₂)
    if are_syntactically_linked(tok_b2, tok_a2):
        return True

    return False


def find_chiasms_in_tokens(tokens, min_distance=1):
    """
    Cherche toutes les paires de lemmes (a, b) telles qu'il existe
    des positions i < j  et  k < l  (avec j <= k ou chevauchement)
    où tokens[i].lemme == a, tokens[j].lemme == b,
       tokens[k].lemme == b, tokens[l].lemme == a
    et a ≠ b, les deux sont des mots de contenu, et au moins une
    des paires (A₁,B₁) ou (B₂,A₂) présente un lien de dépendance
    syntaxique (filtre par analyse en dépendances spaCy).

    Renvoie une liste de dictionnaires décrivant chaque candidat.
    """
    results = []

    # Indexer les positions de chaque lemme (hors stop-lemmes)
    lemme_positions = {}
    for i, tok in enumerate(tokens):
        lem = tok["lemme"]
        if lem in STOP_LEMMES:
            continue
        if len(lem) < MIN_LEMME_LEN:
            continue
        lemme_positions.setdefault(lem, []).append(i)

    # On ne garde que les lemmes qui apparaissent au moins 2 fois
    repeated = {lem: pos for lem, pos in lemme_positions.items() if len(pos) >= 2}

    # Pour chaque paire de lemmes répétés, vérifier le chiasme
    lemmes_list = list(repeated.keys())
    seen = set()

    for idx_a, lem_a in enumerate(lemmes_list):
        for idx_b, lem_b in enumerate(lemmes_list):
            if idx_a >= idx_b:
                continue
            if lem_a == lem_b:
                continue

            # Les deux lemmes doivent être des mots de contenu
            pos_a = {tokens[p]["pos"] for p in repeated[lem_a]}
            pos_b = {tokens[p]["pos"] for p in repeated[lem_b]}
            if not (pos_a & CONTENT_POS) or not (pos_b & CONTENT_POS):
                continue

            positions_a = repeated[lem_a]
            positions_b = repeated[lem_b]

            # Chercher A…B puis B…A
            # A avant B : il existe i dans positions_a, j dans positions_b, i < j
            # B avant A : il existe k dans positions_b, l in positions_a, k < l
            # Et on veut que la seconde paire soit APRÈS la première :
            #   j <= k  (ou au moins k > i)

            for i in positions_a:
                for j in positions_b:
                    if j <= i:
                        continue
                    # distance minimale entre les deux premiers termes
                    if (j - i) < min_distance:
                        continue
                    for k in positions_b:
                        if k <= j:
                            continue
                        if k == j:
                            continue
                        for l in positions_a:
                            if l <= k:
                                continue
                            if (l - k) < min_distance:
                                continue
                            # On a un chiasme A(i)…B(j)…B(k)…A(l)
                            key = (i, j, k, l)
                            if key in seen:
                                continue
                            seen.add(key)
                            # Filtre syntaxique : vérifier qu'au moins
                            # une paire (A₁,B₁) ou (B₂,A₂) est liée
                            if not chiasm_has_syntactic_link(tokens, (i, j, k, l)):
                                continue
                            results.append({
                                "lemme_a": lem_a,
                                "lemme_b": lem_b,
                                "positions": (i, j, k, l),
                                "formes": (
                                    tokens[i]["forme"],
                                    tokens[j]["forme"],
                                    tokens[k]["forme"],
                                    tokens[l]["forme"],
                                ),
                            })
    return results


def detect_antimetaboles(text, nlp, progress_callback=None):
    """
    Fonction principale de détection.
    Renvoie une liste de résultats, chacun contenant le contexte
    et les mots détectés.
    """
    # Traiter le texte par morceaux si très long (limite spaCy)
    max_len = 900_000
    if len(text) > max_len:
        # Découper en blocs avec chevauchement d'une phrase
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_len, len(text))
            chunks.append(text[start:end])
            start = end
    else:
        chunks = [text]

    all_results = []

    for chunk_idx, chunk in enumerate(chunks):
        doc = nlp(chunk)
        sentences = list(doc.sents)
        total = len(sentences)

        for s_idx, sent in enumerate(sentences):
            if progress_callback and s_idx % 50 == 0:
                pct = int((s_idx / max(total, 1)) * 100)
                progress_callback(pct, f"Analyse : phrase {s_idx}/{total}…")

            # --- Détection intra-phrase ---
            tokens = extract_tokens(sent)
            chiasms = find_chiasms_in_tokens(tokens)
            for ch in chiasms:
                all_results.append({
                    "contexte": sent.text.strip(),
                    "lemme_a": ch["lemme_a"],
                    "lemme_b": ch["lemme_b"],
                    "formes": ch["formes"],
                    "sent_idx": s_idx,
                })

    if progress_callback:
        progress_callback(100, "Terminé.")

    # Regrouper par phrase (contexte identique)
    from collections import OrderedDict
    grouped = OrderedDict()
    for r in all_results:
        ctx = r["contexte"]
        if ctx not in grouped:
            grouped[ctx] = {
                "contexte": ctx,
                "sent_idx": r["sent_idx"],
                "motifs": [],
            }
        motif = {
            "lemme_a": r["lemme_a"],
            "lemme_b": r["lemme_b"],
            "formes": r["formes"],
        }
        # Éviter les doublons de motifs dans le même groupe
        motif_key = (r["lemme_a"], r["lemme_b"])
        existing_keys = {(m["lemme_a"], m["lemme_b"]) for m in grouped[ctx]["motifs"]}
        if motif_key not in existing_keys:
            grouped[ctx]["motifs"].append(motif)

    result_list = list(grouped.values())
    result_list.sort(key=lambda r: r["sent_idx"])

    return result_list


# ---------------------------------------------------------------------------
# Interface graphique tkinter
# ---------------------------------------------------------------------------

class AntimetaboleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur d'antimétaboles")
        self.root.geometry("950x700")
        self.root.minsize(700, 500)
        
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
        if os.path.exists(icon_path):
            icon = tk.PhotoImage(file=icon_path)
            self.root.iconphoto(True, icon)
            self._icon = icon  # garder une référence sinon Python libère l'image
        self.nlp = None
        self.results = []

        self._build_ui()

    def _build_ui(self):
        # --- Barre du haut ---
        top = tk.Frame(self.root, padx=10, pady=10)
        top.pack(fill=tk.X)

        tk.Label(top, text="Détecteur d'antimétaboles", font=("sans-serif", 16, "bold")).pack(anchor=tk.W)
        tk.Label(
            top,
            text="Sélectionnez un fichier .txt pour y chercher des structures chiastiques (A…B…B…A).",
            font=("sans-serif", 10),
            fg="#555",
        ).pack(anchor=tk.W, pady=(2, 0))

        # --- Sélection de fichier ---
        file_frame = tk.Frame(self.root, padx=10)
        file_frame.pack(fill=tk.X)

        self.file_var = tk.StringVar(value="(aucun fichier sélectionné)")
        tk.Label(file_frame, text="Fichier :").pack(side=tk.LEFT)
        tk.Label(file_frame, textvariable=self.file_var, fg="#333", width=60, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="Parcourir…", command=self._select_file).pack(side=tk.LEFT)
        self.btn_analyze = tk.Button(file_frame, text="Analyser", command=self._run_analysis, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)

        # --- Barre de progression ---
        prog_frame = tk.Frame(self.root, padx=10, pady=5)
        prog_frame.pack(fill=tk.X)
        self.progress = ttk.Progressbar(prog_frame, length=400, mode="determinate")
        self.progress.pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="En attente.")
        tk.Label(prog_frame, textvariable=self.status_var, fg="#555").pack(side=tk.LEFT, padx=10)

        # --- Compteur de résultats ---
        count_frame = tk.Frame(self.root, padx=10)
        count_frame.pack(fill=tk.X)
        self.count_var = tk.StringVar(value="")
        tk.Label(count_frame, textvariable=self.count_var, font=("sans-serif", 10, "bold")).pack(anchor=tk.W)

        # --- Zone de résultats ---
        result_frame = tk.Frame(self.root, padx=10, pady=5)
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = scrolledtext.ScrolledText(
            result_frame,
            wrap=tk.WORD,
            font=("monospace", 10),
            state=tk.DISABLED,
            bg="#fefefe",
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Tags pour la coloration
        self.result_text.tag_configure("header", font=("sans-serif", 11, "bold"), foreground="#2255aa")
        self.result_text.tag_configure("match", foreground="#cc3300", font=("monospace", 10, "bold"))
        self.result_text.tag_configure("info", foreground="#666666")
        self.result_text.tag_configure("separator", foreground="#cccccc")
        self.result_text.tag_configure("context", foreground="#222222")

        # --- Bouton exporter ---
        bottom = tk.Frame(self.root, padx=10, pady=8)
        bottom.pack(fill=tk.X)
        self.btn_export = tk.Button(bottom, text="Exporter les résultats (.txt)", command=self._export, state=tk.DISABLED)
        self.btn_export.pack(side=tk.RIGHT)

        self.filepath = None

    def _select_file(self):
        path = filedialog.askopenfilename(
            title="Choisir un fichier texte",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous", "*.*")],
        )
        if path:
            self.filepath = path
            self.file_var.set(os.path.basename(path))
            self.btn_analyze.config(state=tk.NORMAL)

    def _run_analysis(self):
        if not self.filepath:
            return
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.DISABLED)
        self.count_var.set("")
        self._clear_results()
        self._set_status(0, "Chargement du modèle spaCy…")
        threading.Thread(target=self._analysis_thread, daemon=True).start()

    def _analysis_thread(self):
        try:
            # Charger le modèle si nécessaire
            if self.nlp is None:
                self.nlp = load_model()
                # Augmenter la limite de longueur pour les gros textes
                self.nlp.max_length = 2_000_000

            self._set_status(5, "Lecture du fichier…")

            # Lire le fichier
            with open(self.filepath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            if not text.strip():
                self.root.after(0, lambda: messagebox.showwarning("Attention", "Le fichier est vide."))
                self._set_status(0, "En attente.")
                self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))
                return

            self._set_status(10, f"Analyse de {len(text):,} caractères…")

            results = detect_antimetaboles(text, self.nlp, progress_callback=self._set_status)
            self.results = results

            # Afficher les résultats dans le fil principal
            self.root.after(0, lambda: self._display_results(results))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erreur", str(e)))
            self._set_status(0, "Erreur.")
        finally:
            self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))

    def _set_status(self, pct, msg):
        self.root.after(0, lambda: self.progress.configure(value=pct))
        self.root.after(0, lambda: self.status_var.set(msg))

    def _clear_results(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)

    def _display_results(self, results):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)

        if not results:
            self.result_text.insert(tk.END, "Aucune antimétabole candidate détectée.\n")
            self.count_var.set("0 résultat.")
        else:
            self.count_var.set(f"{len(results)} phrase(s) avec candidat(s).")
            for i, r in enumerate(results, 1):
                # Numéro
                self.result_text.insert(tk.END, f"── Candidat {i} ──\n", "separator")

                # Motifs détectés
                for m in r["motifs"]:
                    a1, b1, b2, a2 = m["formes"]
                    self.result_text.insert(tk.END, "  Motif : ", "info")
                    self.result_text.insert(tk.END, f"{a1}…{b1}", "match")
                    self.result_text.insert(tk.END, "  →  ", "info")
                    self.result_text.insert(tk.END, f"{b2}…{a2}", "match")
                    self.result_text.insert(tk.END, f"   ({m['lemme_a']}, {m['lemme_b']})\n", "info")

                # Contexte (une seule fois)
                self.result_text.insert(tk.END, "  Contexte : ", "info")
                self.result_text.insert(tk.END, r["contexte"] + "\n", "context")

                self.result_text.insert(tk.END, "\n")

        self.result_text.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.NORMAL if results else tk.DISABLED)

    def _export(self):
        if not self.results:
            return
        path = filedialog.asksaveasfilename(
            title="Enregistrer les résultats",
            defaultextension=".txt",
            filetypes=[("Fichier texte", "*.txt")],
            initialfile="antimetaboles_resultats.txt",
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"DÉTECTION D'ANTIMÉTABOLES\n")
            f.write(f"Fichier analysé : {self.filepath}\n")
            f.write(f"Nombre de candidats : {len(self.results)}\n")
            f.write("=" * 60 + "\n\n")
            for i, r in enumerate(self.results, 1):
                f.write(f"── Candidat {i} ──\n")
                for m in r["motifs"]:
                    a1, b1, b2, a2 = m["formes"]
                    f.write(f"  Motif : {a1}…{b1}  →  {b2}…{a2}   ({m['lemme_a']}, {m['lemme_b']})\n")
                f.write(f"  Contexte : {r['contexte']}\n\n")

        messagebox.showinfo("Export", f"Résultats enregistrés dans :\n{path}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = AntimetaboleApp(root)
    root.mainloop()
