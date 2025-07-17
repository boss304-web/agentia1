#!/usr/bin/env python3
"""
Agent IA Chef de Projet Complet
===============================

Un système intelligent de gestion de projets avec :
- Recherche automatique d'informations
- Gestion complète des projets dans Google Sheets
- Analyses et rappels automatiques
- Interface de chat interactive
- Suivi des stagnations et alertes

Utilise les modèles gratuits OpenRouter et AutoGen local.
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import time
import schedule
import threading
import sys

# AutoGen imports
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Google Sheets imports
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Email/Notifications (optionnel)
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentChefProjet:
    """Agent IA Chef de Projet - Système complet de gestion"""
    
    def _init_(self, config_path: str = "config.json"):
        """
        Initialise l'agent avec la configuration
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.setup_agents()
        self.setup_google_sheets()
        self.projects_cache = {}
        self.last_update = None
        
        # Démarrer le système de surveillance
        self.start_monitoring()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier JSON"""
        default_config = {
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "google_creds_path": os.getenv("GOOGLE_CREDS_PATH", "credentials.json"),
            "sheet_name": os.getenv("SHEET_NAME", "Gestion_Projets"),
            "email_config": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "email": os.getenv("EMAIL_USER", ""),
                "password": os.getenv("EMAIL_PASS", "")
            },
            "monitoring": {
                "check_interval_hours": 24,
                "stagnation_alert_days": 7,
                "urgent_alert_days": 14
            }
        }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Fusionner avec la config par défaut
                default_config.update(config)
                return default_config
        except FileNotFoundError:
            logger.info(f"Config file not found, using default config")
            # Créer le fichier de config par défaut
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def setup_agents(self):
        """Configure les agents IA spécialisés"""
        
        # Configuration commune
        base_config = {
            "api_key": self.config["openrouter_api_key"],
            "base_url": "https://openrouter.ai/api/v1",
            "max_tokens": 1000,
        }
        
        # 🔍 Agent Recherche - Spécialisé dans la recherche et l'analyse
        self.agent_recherche = ConversableAgent(
            name="AgentRecherche",
            system_message="""Tu es un expert en recherche d'informations et analyse de projets.
            
            Tes compétences :
            - Analyser les requêtes de recherche et projets
            - Structurer l'information de manière logique
            - Identifier les points critiques et opportunités
            - Proposer des solutions pratiques
            - Faire des recommandations basées sur les données
            
            Format de réponse :
            ## 🔍 ANALYSE
            [Ton analyse détaillée]
            
            ## 📊 POINTS CLÉS
            [Liste des éléments importants]
            
            ## 🎯 RECOMMANDATIONS
            [Actions concrètes à entreprendre]
            
            ## ⚠ ALERTES
            [Risques ou points d'attention]
            
            Sois précis, factuel et orienté solution.""",
            llm_config={**base_config, "model": "meta-llama/llama-3.1-8b-instruct:free", "temperature": 0.3},
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )
        
        # 📝 Agent Rédacteur - Spécialisé dans la synthèse et communication
        self.agent_redacteur = ConversableAgent(
            name="AgentRedacteur",
            system_message="""Tu es un assistant personnel expert en communication et gestion de projets.
            
            Ton rôle :
            - Reformuler les analyses en langage clair et actionnable
            - Prioriser les informations selon leur importance
            - Créer des résumés engageants et motivants
            - Proposer des plans d'action concrets
            - Adapter le ton selon le contexte (urgent, informatif, encourageant)
            
            Tu dois toujours :
            - Commencer par un résumé exécutif
            - Utiliser des emojis pour structurer
            - Être bienveillant mais direct
            - Proposer des actions concrètes
            - Inclure des échéances si pertinent
            
            Format de réponse :
            ## 🎯 RÉSUMÉ EXÉCUTIF
            [Synthèse en 2-3 phrases]
            
            ## 📋 PLAN D'ACTION
            [Étapes concrètes à suivre]
            
            ## 🔔 RAPPELS
            [Ce qu'il faut retenir]""",
            llm_config={**base_config, "model": "microsoft/wizardlm-2-8x22b:free", "temperature": 0.7},
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )
        
        # 🤖 Agent Planificateur - Spécialisé dans la planification et suivi
        self.agent_planificateur = ConversableAgent(
            name="AgentPlanificateur",
            system_message="""Tu es un expert en planification et suivi de projets.
            
            Tes spécialités :
            - Analyser les deadlines et priorités
            - Détecter les blocages et retards
            - Proposer des réajustements de planning
            - Optimiser l'allocation des ressources
            - Anticiper les risques
            
            Tu dois toujours :
            - Évaluer la faisabilité des délais
            - Identifier les dépendances entre tâches
            - Proposer des alternatives en cas de blocage
            - Calculer les impacts sur le planning global
            
            Format de réponse :
            ## 📅 ANALYSE PLANNING
            [Évaluation des délais et contraintes]
            
            ## 🚨 ALERTES TIMING
            [Deadlines critiques ou retards]
            
            ## 🔄 RÉAJUSTEMENTS
            [Propositions d'optimisation]""",
            llm_config={**base_config, "model": "google/gemma-2-9b-it:free", "temperature": 0.4},
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )
        
        logger.info("✅ Agents IA configurés avec succès")
    
    def setup_google_sheets(self):
        """Configure la connexion à Google Sheets"""
        try:
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                self.config["google_creds_path"], scope
            )
            
            self.gc = gspread.authorize(creds)
            self.sheet = self.gc.open(self.config["sheet_name"]).sheet1
            
            # Configurer les en-têtes avec votre structure
            self.setup_sheet_headers()
            
            logger.info("✅ Connexion Google Sheets établie")
            
        except Exception as e:
            logger.error(f"❌ Erreur Google Sheets : {e}")
            raise
    
    def setup_sheet_headers(self):
        """Configure les en-têtes selon votre structure"""
        headers = [
            'Nom_Projet', 'Statut', 'Priorité', 'Description', 'Prochaine_Action',
            'Deadline', 'Date_Creation', 'Dernière_MAJ', 'Jours_Stagnation',
            'Alerte', 'Progression_%', 'Notes'
        ]
        
        try:
            existing_headers = self.sheet.row_values(1)
            if not existing_headers or existing_headers != headers:
                self.sheet.clear()
                self.sheet.insert_row(headers, 1)
                logger.info("📋 En-têtes configurés")
        except Exception as e:
            logger.error(f"❌ Erreur en-têtes : {e}")
    
    def rechercher_information(self, query: str, project_context: str = "") -> Dict[str, Any]:
        """
        Effectue une recherche d'information avec analyse par les agents
        
        Args:
            query: Requête de recherche
            project_context: Contexte du projet (optionnel)
            
        Returns:
            Résultat de la recherche avec analyse
        """
        try:
            logger.info(f"🔍 Recherche en cours : {query}")
            
            # Préparer le contexte complet
            context = f"""
            Requête de recherche : {query}
            Contexte du projet : {project_context}
            Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Analyse cette requête et fournis une recherche approfondie.
            """
            
            # Étape 1 : Analyse par l'agent recherche
            recherche_result = self.agent_recherche.generate_reply(
                messages=[{"role": "user", "content": context}]
            )
            
            # Étape 2 : Planification si nécessaire
            if any(word in query.lower() for word in ['deadline', 'planning', 'délai', 'échéance']):
                planning_result = self.agent_planificateur.generate_reply(
                    messages=[{"role": "user", "content": f"Analyse planning pour : {recherche_result}"}]
                )
            else:
                planning_result = ""
            
            # Étape 3 : Synthèse finale
            synthesis_context = f"""
            Recherche effectuée : {query}
            
            Analyse de recherche :
            {recherche_result}
            
            Analyse planning :
            {planning_result}
            
            Synthétise cette information pour l'utilisateur.
            """
            
            final_result = self.agent_redacteur.generate_reply(
                messages=[{"role": "user", "content": synthesis_context}]
            )
            
            # Sauvegarder la recherche
            self.save_research_log(query, final_result, project_context)
            
            return {
                "query": query,
                "context": project_context,
                "raw_analysis": recherche_result,
                "planning_analysis": planning_result,
                "final_synthesis": final_result,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche : {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def ajouter_projet(self, nom_projet: str, description: str, priorite: int = 2, 
                      deadline: str = "", notes: str = "") -> Dict[str, Any]:
        """
        Ajoute un nouveau projet au système
        
        Args:
            nom_projet: Nom du projet
            description: Description détaillée
            priorite: Priorité (1=Haute, 2=Moyenne, 3=Basse, 4=Optionnelle)
            deadline: Date limite (format YYYY-MM-DD)
            notes: Notes additionnelles
            
        Returns:
            Résultat de l'ajout
        """
        try:
            # Analyser le projet avec l'agent recherche
            project_analysis = self.agent_recherche.generate_reply(
                messages=[{"role": "user", "content": f"""
                Analyse ce nouveau projet :
                
                Nom : {nom_projet}
                Description : {description}
                Priorité : {priorite}
                Deadline : {deadline}
                Notes : {notes}
                
                Propose une première action et identifie les risques potentiels.
                """}]
            )
            
            # Obtenir des recommandations de planification
            planning_advice = self.agent_planificateur.generate_reply(
                messages=[{"role": "user", "content": f"""
                Nouveau projet à planifier :
                {project_analysis}
                
                Évalue la faisabilité du délai et propose un plan d'action.
                """}]
            )
            
            # Synthèse finale
            project_summary = self.agent_redacteur.generate_reply(
                messages=[{"role": "user", "content": f"""
                Synthétise l'analyse de ce nouveau projet :
                
                Analyse : {project_analysis}
                Planning : {planning_advice}
                
                Fournis un résumé actionnable pour démarrer le projet.
                """}]
            )
            
            # Préparer les données pour Google Sheets
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Extraire la première action recommandée
            prochaine_action = "Analyser les besoins"  # Valeur par défaut
            if "action" in project_analysis.lower():
                lines = project_analysis.split('\n')
                for line in lines:
                    if "action" in line.lower() and len(line) > 10:
                        prochaine_action = line.strip()[:100]
                        break
            
            row_data = [
                nom_projet,
                "À faire",
                priorite,
                description,
                prochaine_action,
                deadline if deadline else "",
                current_date,
                current_date,
                0,  # Jours_Stagnation
                "",  # Alerte
                0,   # Progression_%
                notes + f"\n\nAnalyse IA : {project_summary[:200]}..."
            ]
            
            # Ajouter à la feuille
            self.sheet.append_row(row_data)
            
            # Mettre à jour le cache
            self.refresh_projects_cache()
            
            logger.info(f"✅ Projet ajouté : {nom_projet}")
            
            return {
                "nom_projet": nom_projet,
                "status": "success",
                "analysis": project_analysis,
                "planning": planning_advice,
                "summary": project_summary,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur ajout projet : {e}")
            return {
                "nom_projet": nom_projet,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def mettre_a_jour_projet(self, nom_projet: str, **kwargs) -> Dict[str, Any]:
        """
        Met à jour un projet existant
        
        Args:
            nom_projet: Nom du projet à mettre à jour
            **kwargs: Champs à mettre à jour
            
        Returns:
            Résultat de la mise à jour
        """
        try:
            # Trouver le projet
            projects = self.get_all_projects()
            project_row = None
            
            for i, project in enumerate(projects, 2):  # Ligne 2 = premier projet
                if project.get('Nom_Projet') == nom_projet:
                    project_row = i
                    break
            
            if not project_row:
                return {
                    "nom_projet": nom_projet,
                    "status": "error",
                    "error": "Projet non trouvé"
                }
            
            # Colonnes mapping
            columns = {
                'statut': 2, 'priorite': 3, 'description': 4, 'prochaine_action': 5,
                'deadline': 6, 'notes': 12
            }
            
            # Mettre à jour les champs
            for field, value in kwargs.items():
                if field.lower() in columns:
                    self.sheet.update_cell(project_row, columns[field.lower()], value)
            
            # Mettre à jour la date de dernière MAJ
            self.sheet.update_cell(project_row, 8, datetime.now().strftime("%Y-%m-%d"))
            
            # Analyser la mise à jour
            update_context = f"""
            Projet mis à jour : {nom_projet}
            Modifications : {kwargs}
            
            Analyse cette mise à jour et fournis des recommandations.
            """
            
            update_analysis = self.agent_redacteur.generate_reply(
                messages=[{"role": "user", "content": update_context}]
            )
            
            logger.info(f"✅ Projet mis à jour : {nom_projet}")
            
            return {
                "nom_projet": nom_projet,
                "status": "success",
                "updates": kwargs,
                "analysis": update_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour : {e}")
            return {
                "nom_projet": nom_projet,
                "status": "error",
                "error": str(e)
            }
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Récupère tous les projets"""
        try:
            return self.sheet.get_all_records()
        except Exception as e:
            logger.error(f"❌ Erreur récupération projets : {e}")
            return []
    
    def rechercher_projet(self, query: str) -> List[Dict[str, Any]]:
        """
        Recherche un projet spécifique
        
        Args:
            query: Terme de recherche
            
        Returns:
            Liste des projets correspondants
        """
        projects = self.get_all_projects()
        results = []
        
        query_lower = query.lower()
        
        for project in projects:
            # Rechercher dans le nom, description et notes
            if (query_lower in project.get('Nom_Projet', '').lower() or
                query_lower in project.get('Description', '').lower() or
                query_lower in project.get('Notes', '').lower()):
                results.append(project)
        
        return results
    
    def generer_rapport_quotidien(self) -> Dict[str, Any]:
        """Génère un rapport quotidien des projets"""
        try:
            projects = self.get_all_projects()
            
            # Préparer les statistiques
            stats = {
                'total': len(projects),
                'en_cours': len([p for p in projects if p.get('Statut') == 'En cours']),
                'a_faire': len([p for p in projects if p.get('Statut') == 'À faire']),
                'termines': len([p for p in projects if p.get('Statut') == 'Terminé']),
                'bloques': len([p for p in projects if p.get('Statut') == 'Bloqué']),
                'urgents': len([p for p in projects if '🚨' in p.get('Alerte', '')]),
            }
            
            # Projets critiques
            critiques = [p for p in projects if p.get('Priorité') == 1 and p.get('Statut') != 'Terminé']
            
            # Projets en retard
            today = datetime.now().date()
            retards = []
            for p in projects:
                if p.get('Deadline') and p.get('Statut') != 'Terminé':
                    try:
                        deadline = datetime.strptime(p['Deadline'], '%Y-%m-%d').date()
                        if deadline < today:
                            retards.append(p)
                    except ValueError:
                        pass
            
            # Générer l'analyse avec l'agent
            rapport_context = f"""
            Génère un rapport quotidien des projets :
            
            Statistiques :
            - Total : {stats['total']} projets
            - En cours : {stats['en_cours']}
            - À faire : {stats['a_faire']}
            - Terminés : {stats['termines']}
            - Bloqués : {stats['bloques']}
            - Urgents : {stats['urgents']}
            
            Projets critiques : {len(critiques)}
            Projets en retard : {len(retards)}
            
            Fournis un résumé avec recommandations d'actions.
            """
            
            rapport_analysis = self.agent_redacteur.generate_reply(
                messages=[{"role": "user", "content": rapport_context}]
            )
            
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "statistics": stats,
                "projets_critiques": critiques,
                "projets_retard": retards,
                "analysis": rapport_analysis,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur rapport quotidien : {e}")
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "status": "error",
                "error": str(e)
            }
    
    def save_research_log(self, query: str, result: str, context: str = ""):
        """Sauvegarde les recherches dans un log"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "context": context,
                "result": result[:500] + "..." if len(result) > 500 else result
            }
            
            # Sauvegarder dans un fichier JSON
            log_file = "research_log.json"
            logs = []
            
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            # Garder seulement les 100 dernières recherches
            if len(logs) > 100:
                logs = logs[-100:]
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde log : {e}")
    
    def refresh_projects_cache(self):
        """Met à jour le cache des projets"""
        try:
            self.projects_cache = self.get_all_projects()
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"❌ Erreur refresh cache : {e}")
    
    def start_monitoring(self):
        """Démarre le système de surveillance automatique"""
        def monitor_projects():
            try:
                # Vérifier les projets stagnants
                projects = self.get_all_projects()
                alerts = []
                
                for project in projects:
                    if project.get('Statut') not in ['Terminé', 'Annulé']:
                        # Vérifier la stagnation
                        jours_stagnation = project.get('Jours_Stagnation', 0)
                        if isinstance(jours_stagnation, (int, float)) and jours_stagnation > 7:
                            alerts.append(f"⚠ {project['Nom_Projet']} : {jours_stagnation} jours sans mise à jour")
                
                if alerts:
                    alert_message = "\n".join(alerts)
                    logger.warning(f"🚨 Alertes détectées :\n{alert_message}")
                    
                    # Optionnel : envoyer par email
                    # self.send_alert_email(alert_message)
                
            except Exception as e:
                logger.error(f"❌ Erreur monitoring : {e}")
        
        # Programmer les vérifications
        schedule.every(24).hours.do(monitor_projects)
        
        # Lancer le thread de surveillance
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Vérifier toutes les heures
        
        monitoring_thread = threading.Thread(target=run_scheduler, daemon=True)
        monitoring_thread.start()
        
        logger.info("🔄 Système de surveillance démarré")
    
    def chat_interface(self):
        """Interface de chat interactive"""
        print("\n" + "="*60)
        print("🤖 AGENT IA CHEF DE PROJET")
        print("="*60)
        print("Commandes disponibles :")
        print("- rechercher [query] : Rechercher des informations")
        print("- ajouter : Ajouter un nouveau projet")
        print("- projet [nom] : Chercher un projet spécifique")
        print("- rapport : Générer un rapport quotidien")
        print("- projets : Lister tous les projets")
        print("- quit : Quitter")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🎯 Votre demande : ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Au revoir !")
                    break
                
                elif user_input.lower().startswith('rechercher '):
                    query = user_input[11:]
                    print(f"\n🔍 Recherche en cours : {query}")
                    result = self.rechercher_information(query)
                    if result['status'] == 'success':
                        print(f"\n📊 Résultat :\n{result['final_synthesis']}")
                    else:
                        print(f"\n❌ Erreur : {result['error']}")
                
                elif user_input.lower() == 'ajouter':
                    print("\n📝 Ajout d'un nouveau projet")
                    nom = input("Nom du projet : ")
                    description = input("Description : ")
                    priorite = input("Priorité (1-4) : ")
                    deadline = input("Deadline (YYYY-MM-DD) : ")
                    notes = input("Notes : ")
                    
                    result = self.ajouter_projet(
                        nom, description, 
                        int(priorite) if priorite.isdigit() else 2,
                        deadline, notes
                    )
                    
                    if result['status'] == 'success':
                        print(f"\n✅ Projet ajouté avec succès !")
                        print(f"📋 Analyse :\n{result['summary']}")
                    else:
                        print(f"\n❌ Erreur : {result['error']}")
                
                elif user_input.lower().startswith('projet '):
                    nom_projet = user_input[7:]
                    results = self.rechercher_projet(nom_projet)
                    
                    if results:
                        print(f"\n📋 Projets trouvés ({len(results)}) :")
                        for i, projet in enumerate(results, 1):
                            print(f"{i}. {projet['Nom_Projet']} - {projet['Statut']}")
                            print(f"   📝 {projet['Description']}")
                            print(f"   📅 Deadline: {projet.get('Deadline', 'N/A')}")
                            print(f"   🔄 Prochaine action: {projet.get('Prochaine_Action', 'N/A')}")
                            print()
                    else:
                        print(f"\n❌ Aucun projet trouvé pour '{nom_projet}'")
                
                elif user_input.lower() == 'rapport':
                    print("\n📊 Génération du rapport quotidien...")
                    rapport = self.generer_rapport_quotidien()
                    
                    if rapport['status'] == 'success':
                        print(f"\n📈 RAPPORT DU {rapport['date']}")
                        print(f"📊 Statistiques : {rapport['statistics']}")
                        print(f"\n🎯 Analyse :\n{rapport['analysis']}")
                    else:
                        print(f"\n❌ Erreur : {rapport['error']}")
                
                elif user_input.lower() == 'projets':
                    projects = self.get_all_projects()
                    print(f"\n📋 Tous les projets ({len(projects)}) :")
                    for i, projet in enumerate(projects, 1):
                        status_emoji = {"En cours": "🔄", "À faire": "📋", "Terminé": "✅", "Bloqué": "🚫"}.get(projet['Statut'], "❓")
                        print(f"{i}. {status_emoji} {projet['Nom_Projet']} - {projet['Statut']}")
                
                else:
                    # Requête libre - analyser avec l'agent
                    print(f"\n🤖 Traitement de votre demande...")
                    result = self.agent_redacteur.generate_reply(
                        messages=[{"role": "user", "content": f"""
                        L'utilisateur demande : {user_input}
                        
                        Contexte : Tu es l'assistant chef de projet. L'utilisateur peut :
                        - Poser des questions sur ses projets
                        - Demander des conseils en gestion de projet
                        - Chercher de l'aide pour organiser son travail
                        - Demander des analyses ou recommandations
                        
                        Projets actuels : {len(self.get_all_projects())} projets en cours
                        
                        Réponds de manière utile et actionnable.
                        """}]
                    )
                    print(f"\n💡 Réponse :\n{result}")
                
            except KeyboardInterrupt:
                print("\n👋 Au revoir !")
                break
            except Exception as e:
                print(f"\n❌ Erreur : {e}")
                logger.error(f"Erreur dans le chat : {e}")


def main():
    """Fonction principale"""
    print("🚀 Initialisation de l'Agent IA Chef de Projet...")
    
    try:
        # Initialiser l'agent
        agent = AgentChefProjet()
        
        # Démarrer l'interface de chat
        agent.chat_interface()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation : {e}")
        logger.error(f"Erreur principale : {e}")


def create_sample_config():
    """Crée un fichier de configuration d'exemple"""
    config = {
        "openrouter_api_key": "YOUR_OPENROUTER_API_KEY",
        "google_creds_path": "path/to/your/google-credentials.json",
        "sheet_name": "Gestion_Projets",
        "email_config": {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email": "your_email@gmail.com",
            "password": "your_app_password"
        },
        "monitoring": {
            "check_interval_hours": 24,
            "stagnation_alert_days": 7,
            "urgent_alert_days": 14
        }
    }
    
    with open("config_example.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("📝 Fichier config_example.json créé")


# Fonctions utilitaires supplémentaires
class ProjectAnalyzer:
    """Classe pour analyses avancées des projets"""
    
    def _init_(self, agent_chef_projet):
        self.agent = agent_chef_projet
    
    def analyze_project_health(self, project_name: str) -> Dict[str, Any]:
        """Analyse la santé d'un projet"""
        projects = self.agent.rechercher_projet(project_name)
        
        if not projects:
            return {"error": "Projet non trouvé"}
        
        project = projects[0]
        
        # Calculer les métriques
        health_score = 100
        issues = []
        
        # Vérifier la stagnation
        jours_stagnation = project.get('Jours_Stagnation', 0)
        if isinstance(jours_stagnation, (int, float)):
            if jours_stagnation > 14:
                health_score -= 30
                issues.append(f"🚨 Stagnation critique: {jours_stagnation} jours")
            elif jours_stagnation > 7:
                health_score -= 15
                issues.append(f"⚠ Stagnation modérée: {jours_stagnation} jours")
        
        # Vérifier le statut
        if project.get('Statut') == 'Bloqué':
            health_score -= 40
            issues.append("🚫 Projet bloqué")
        
        # Vérifier la deadline
        if project.get('Deadline'):
            try:
                deadline = datetime.strptime(project['Deadline'], '%Y-%m-%d').date()
                today = datetime.now().date()
                days_remaining = (deadline - today).days
                
                if days_remaining < 0:
                    health_score -= 50
                    issues.append(f"📅 En retard de {abs(days_remaining)} jours")
                elif days_remaining <= 3:
                    health_score -= 20
                    issues.append(f"⏰ Deadline dans {days_remaining} jours")
            except ValueError:
                pass
        
        # Vérifier la progression
        progression = project.get('Progression_%', 0)
        if isinstance(progression, (int, float)):
            if progression < 25 and project.get('Statut') == 'En cours':
                health_score -= 10
                issues.append("📊 Progression faible")
        
        # Déterminer le niveau de santé
        if health_score >= 80:
            health_level = "🟢 Excellente"
        elif health_score >= 60:
            health_level = "🟡 Bonne"
        elif health_score >= 40:
            health_level = "🟠 Préoccupante"
        else:
            health_level = "🔴 Critique"
        
        return {
            "project_name": project_name,
            "health_score": max(0, health_score),
            "health_level": health_level,
            "issues": issues,
            "recommendations": self.generate_recommendations(project, issues)
        }
    
    def generate_recommendations(self, project: Dict, issues: List[str]) -> List[str]:
        """Génère des recommandations basées sur les problèmes détectés"""
        recommendations = []
        
        if any("stagnation" in issue.lower() for issue in issues):
            recommendations.append("📞 Planifier une réunion d'équipe pour débloquer le projet")
            recommendations.append("🔄 Revoir les objectifs et priorités")
        
        if any("bloqué" in issue.lower() for issue in issues):
            recommendations.append("🚫 Identifier les blocages et les ressources nécessaires")
            recommendations.append("🤝 Escalader auprès des parties prenantes")
        
        if any("retard" in issue.lower() for issue in issues):
            recommendations.append("⏱ Revoir le planning et les livrables")
            recommendations.append("📊 Prioriser les tâches critiques")
        
        if any("progression" in issue.lower() for issue in issues):
            recommendations.append("🎯 Décomposer en sous-tâches plus petites")
            recommendations.append("📈 Définir des jalons intermédiaires")
        
        return recommendations


class ReportGenerator:
    """Générateur de rapports avancés"""
    
    def _init_(self, agent_chef_projet):
        self.agent = agent_chef_projet
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """Génère un rapport hebdomadaire"""
        projects = self.agent.get_all_projects()
        
        # Analyse des tendances
        active_projects = [p for p in projects if p.get('Statut') in ['En cours', 'À faire']]
        completed_this_week = [p for p in projects if p.get('Statut') == 'Terminé']
        
        # Calcul des métriques
        total_progression = sum(p.get('Progression_%', 0) for p in active_projects)
        avg_progression = total_progression / len(active_projects) if active_projects else 0
        
        # Projets à risque
        at_risk_projects = []
        for project in active_projects:
            analyzer = ProjectAnalyzer(self.agent)
            health = analyzer.analyze_project_health(project['Nom_Projet'])
            if health.get('health_score', 100) < 60:
                at_risk_projects.append(project)
        
        return {
            "periode": f"Semaine du {datetime.now().strftime('%Y-%m-%d')}",
            "total_projects": len(projects),
            "active_projects": len(active_projects),
            "completed_projects": len(completed_this_week),
            "average_progression": round(avg_progression, 1),
            "at_risk_projects": len(at_risk_projects),
            "at_risk_details": at_risk_projects[:5],  # Top 5 des projets à risque
            "key_metrics": {
                "completion_rate": round(len(completed_this_week) / len(projects) * 100, 1) if projects else 0,
                "efficiency_score": round(avg_progression / len(active_projects) * 100, 1) if active_projects else 0
            }
        }
    
    def export_to_markdown(self, report_data: Dict[str, Any]) -> str:
        """Exporte un rapport en format Markdown"""
        md_content = f"""# Rapport de Gestion de Projets

## 📊 Vue d'ensemble

- *Période*: {report_data.get('periode', 'N/A')}
- *Total des projets*: {report_data.get('total_projects', 0)}
- *Projets actifs*: {report_data.get('active_projects', 0)}
- *Projets terminés*: {report_data.get('completed_projects', 0)}
- *Progression moyenne*: {report_data.get('average_progression', 0)}%

## 🚨 Projets à risque

{report_data.get('at_risk_projects', 0)} projets nécessitent une attention particulière.

## 📈 Métriques clés

- *Taux de completion*: {report_data.get('key_metrics', {}).get('completion_rate', 0)}%
- *Score d'efficacité*: {report_data.get('key_metrics', {}).get('efficiency_score', 0)}%

---
Rapport généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return md_content


# Script d'installation automatique
def install_dependencies():
    """Installe les dépendances nécessaires"""
    import subprocess
    import sys
    
    dependencies = [
        "autogen-agentchat",
        "gspread",
        "oauth2client",
        "schedule",
        "requests"
    ]
    
    print("📦 Installation des dépendances...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installé")
        except subprocess.CalledProcessError:
            print(f"❌ Erreur installation {dep}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent IA Chef de Projet")
    parser.add_argument("--install", action="store_true", help="Installer les dépendances")
    parser.add_argument("--config", action="store_true", help="Créer un fichier de configuration d'exemple")
    parser.add_argument("--chat", action="store_true", help="Démarrer l'interface de chat")
    
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
    elif args.config:
        create_sample_config()
    elif args.chat or len(sys.argv) == 1:
        main()
    else:
        parser.print_help()