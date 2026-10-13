// Carte des sens : les questions de la FAQ, projetées en deux dimensions pour voir les sujets proches se regrouper.
// Lancement : npm run carte (écrit carte-des-sens.svg à la racine du projet)
import { readFile, writeFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";

// 1. Les questions de la FAQ, et un thème pour chacune, qui donnera la couleur du point
const faq = await readFile("corpus/faq/faq-rh.md", "utf8");
const questions = faq.split("\n## ").slice(1).map((bloc) => bloc.split("\n")[0].trim());
const THEMES: [string, RegExp, string][] = [
  ["Télétravail", /télétravail|domicile|jours fixes/i, "#2563eb"],
  ["Congés et absences", /congé|RTT|absence|malade|mariage|PACS|naissance|déménagement/i, "#16a34a"],
  ["Frais et transport", /frais|repas|hôtel|trajets|transport|vélo|titres-restaurant/i, "#d97706"],
  ["Paie et mutuelle", /paie|mutuelle|conjoint|dispensé/i, "#9333ea"],
  ["Autres", /./, "#64748b"],
];
const themeDe = (question: string) => THEMES.find(([, motif]) => motif.test(question))!;

// 2. Un vecteur par question, calculé par le modèle d'embedding local de .env
const extraire = await pipeline("feature-extraction", process.env.EMBEDDING_MODEL!, { dtype: "q8" });
const tenseur = await extraire(questions.map((q) => "query: " + q), { pooling: "mean", normalize: true });
const vecteurs: number[][] = tenseur.tolist();
// …