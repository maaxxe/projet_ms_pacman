##
# @file      Makefile
# @brief     Automatisation du projet DQN MsPacman.
# @details   Cibles disponibles :
#              make train        -> Lance l'entrainement DQN
#              make test         -> Lance une partie visuelle (epsilon=0)
#              make plot         -> Affiche les courbes d'entrainement
#              make record       -> Enregistre les meilleures parties en MP4
#              make doc          -> Genere la documentation Doxygen HTML
#              make doc-open     -> Genere + ouvre dans le navigateur
#              make status       -> Affiche stats entrainement actuel
#              make backup       -> Sauvegarde checkpoint avec timestamp
#              make list-backups -> Liste les checkpoints disponibles
#              make check-env    -> Verifie Python, CUDA, dependances
#              make install      -> Installe les dependances pip
#              make clean        -> Supprime checkpoint + log
#              make clean-videos -> Supprime les MP4
#              make clean-doc    -> Supprime docs/ et Doxyfile
#              make clean-all    -> Supprime tout (avec confirmation)
#              make help         -> Liste toutes les cibles
#

# =============================================================================
# CONFIGURATION
# =============================================================================

## @brief Interpreteur Python
PYTHON      := python

## @brief Dossier racine du projet
PROJECT_DIR := $(shell pwd)

## @brief Fichiers principaux
TRAIN_SCRIPT  := train.py
TEST_SCRIPT   := test_dqn.py
PLOT_SCRIPT   := plot_progress.py
RECORD_SCRIPT := record_best.py

## @brief Dossiers generes
CKPT_DIR  := checkpoints
VIDEO_DIR := videos
DOC_DIR   := docs
LOG_FILE  := log.json

## @brief Fichier de config Doxygen
DOXYFILE  := Doxyfile

# Couleurs terminal
RED    := \033[31m
GREEN  := \033[32m
YELLOW := \033[33m
BLUE   := \033[34m
CYAN   := \033[36m
RESET  := \033[0m

# =============================================================================
# CIBLE PAR DEFAUT
# =============================================================================

.DEFAULT_GOAL := help

# =============================================================================
# ENTRAINEMENT
# =============================================================================

##
# @brief  Lance l'entrainement DQN (reprend depuis checkpoint si existant).
# @details Utilise train.py avec les flags USE_DOUBLE_DQN, USE_DUELING_DQN,
#          USE_PER definis en haut de train.py. Ctrl+C sauvegarde proprement.
#
.PHONY: train
train:
	@echo "$(GREEN)[TRAIN] Lancement entrainement DQN MsPacman...$(RESET)"
	@echo "$(CYAN)        Flags actifs : DoubleDQN / Dueling / PER$(RESET)"
	@echo "$(CYAN)        Checkpoint   : $(CKPT_DIR)/mspacman_dqn.pth$(RESET)"
	@echo "$(YELLOW)        Ctrl+C pour arreter proprement et sauvegarder$(RESET)"
	@echo ""
	$(PYTHON) $(TRAIN_SCRIPT)

# =============================================================================
# TEST ET VISUALISATION
# =============================================================================

##
# @brief  Lance une partie visuelle avec l'agent entraine (epsilon=0).
# @details Charge le checkpoint et joue en mode greedy pur.
#          Affiche score, dots et fantomes en fin de partie.
#
.PHONY: test
test: _check_checkpoint
	@echo "$(GREEN)[TEST] Lancement partie test (epsilon=0)...$(RESET)"
	@echo "$(CYAN)       Modele : $(CKPT_DIR)/mspacman_dqn.pth$(RESET)"
	@echo ""
	$(PYTHON) $(TEST_SCRIPT)

##
# @brief  Affiche les courbes d'entrainement depuis log.json.
# @details Trace 8 graphiques : score, moyenne, epsilon, loss,
#          steps/episode, buffer, dots, fantomes.
#
.PHONY: plot
plot: _check_log
	@echo "$(GREEN)[PLOT] Affichage progression entrainement...$(RESET)"
	@echo "$(CYAN)       Source : $(LOG_FILE)$(RESET)"
	@echo ""
	$(PYTHON) $(PLOT_SCRIPT)

##
# @brief  Enregistre les meilleures parties en MP4 (min 159 dots).
# @details Lance 50 parties, sauvegarde uniquement celles >= min_dots.
#          Sorties dans videos/.
#
.PHONY: record
record: _check_checkpoint
	@echo "$(GREEN)[RECORD] Enregistrement meilleures parties...$(RESET)"
	@echo "$(CYAN)         Sortie : $(VIDEO_DIR)/$(RESET)"
	@echo ""
	$(PYTHON) $(RECORD_SCRIPT)

# =============================================================================
# DOCUMENTATION DOXYGEN
# =============================================================================

##
# @brief  Genere la documentation HTML depuis les commentaires Doxygen.
# @details Cree le Doxyfile si absent, puis lance doxygen.
#          Resultat dans docs/html/index.html.
#
.PHONY: doc
doc: _check_doxygen $(DOXYFILE)
	@echo "$(GREEN)[DOC] Generation documentation Doxygen...$(RESET)"
	doxygen $(DOXYFILE)
	@echo ""
	@echo "$(GREEN)[DOC] Documentation disponible : $(DOC_DIR)/html/index.html$(RESET)"

##
# @brief  Genere la documentation puis l'ouvre dans le navigateur.
#
.PHONY: doc-open
doc-open: doc
	@echo "$(CYAN)[DOC] Ouverture documentation...$(RESET)"
	@xdg-open $(DOC_DIR)/html/index.html 2>/dev/null || \
	 open     $(DOC_DIR)/html/index.html 2>/dev/null || \
	 echo "$(YELLOW)Ouvrez manuellement : $(DOC_DIR)/html/index.html$(RESET)"

##
# @brief  Cree le fichier Doxyfile configure pour Python s'il n'existe pas.
#
$(DOXYFILE):
	@echo "$(YELLOW)[DOC] Creation du Doxyfile...$(RESET)"
	@printf "PROJECT_NAME           = \"DQN MsPacman\"\n"                    >  $(DOXYFILE)
	@printf "PROJECT_BRIEF          = \"Agent DQN pour MsPacman (Atari)\"\n" >> $(DOXYFILE)
	@printf "PROJECT_NUMBER         = 2.1\n"                                  >> $(DOXYFILE)
	@printf "OUTPUT_DIRECTORY       = $(DOC_DIR)\n"                           >> $(DOXYFILE)
	@printf "INPUT                  = .\n"                                    >> $(DOXYFILE)
	@printf "FILE_PATTERNS          = *.py\n"                                 >> $(DOXYFILE)
	@printf "RECURSIVE              = NO\n"                                   >> $(DOXYFILE)
	@printf "GENERATE_HTML          = YES\n"                                  >> $(DOXYFILE)
	@printf "GENERATE_LATEX         = NO\n"                                   >> $(DOXYFILE)
	@printf "EXTRACT_ALL            = YES\n"                                  >> $(DOXYFILE)
	@printf "EXTRACT_PRIVATE        = YES\n"                                  >> $(DOXYFILE)
	@printf "EXTRACT_STATIC         = YES\n"                                  >> $(DOXYFILE)
	@printf "HAVE_DOT               = NO\n"                                   >> $(DOXYFILE)
	@printf "OPTIMIZE_OUTPUT_JAVA   = YES\n"                                  >> $(DOXYFILE)
	@printf "PYTHON_DOCSTRING       = YES\n"                                  >> $(DOXYFILE)
	@printf "HTML_OUTPUT            = html\n"                                 >> $(DOXYFILE)
	@printf "HTML_COLORSTYLE_HUE    = 220\n"                                  >> $(DOXYFILE)
	@printf "GENERATE_TREEVIEW      = YES\n"                                  >> $(DOXYFILE)
	@printf "QUIET                  = YES\n"                                  >> $(DOXYFILE)
	@echo "$(GREEN)[DOC] Doxyfile cree$(RESET)"

# =============================================================================
# UTILITAIRES
# =============================================================================

##
# @brief  Affiche les statistiques de l'entrainement en cours.
# @details Lit log.json et affiche : episodes, best score, last score,
#          last loss, epsilon, avg10, taille du checkpoint.
#
.PHONY: status
status: _check_log
	@echo "$(GREEN)[STATUS] Statistiques entrainement MsPacman$(RESET)"
	@echo "$(CYAN)--------------------------------------------$(RESET)"
	@$(PYTHON) -c "\
import json, os; \
data = json.load(open('$(LOG_FILE)')); \
keys = sorted(data.keys(), key=lambda k: int(k.split('_')[-1])); \
n = len(keys); \
scores = [data[k].get('score', 0) for k in keys]; \
last = data[keys[-1]] if keys else {}; \
print(f'  Episodes    : {n}'); \
print(f'  Best score  : {max(scores):.0f}'); \
print(f'  Last score  : {last.get(\"score\", 0):.0f}'); \
print(f'  Last loss   : {last.get(\"loss\", 0):.4f}'); \
print(f'  Last eps    : {last.get(\"eps\", last.get(\"epsilon\", 0)):.4f}'); \
print(f'  Avg (10)    : {last.get(\"avg10\", last.get(\"avg_score_10\", 0)):.1f}'); \
ckpt = '$(CKPT_DIR)/mspacman_dqn.pth'; \
size = os.path.getsize(ckpt)/1e6 if os.path.exists(ckpt) else 0; \
print(f'  Checkpoint  : {size:.1f} MB'); \
"
	@echo "$(CYAN)--------------------------------------------$(RESET)"

##
# @brief  Sauvegarde le checkpoint actuel avec un timestamp horodate.
# @details Copie mspacman_dqn.pth vers mspacman_dqn_backup_YYYYMMDD_HHMMSS.pth
#
.PHONY: backup
backup: _check_checkpoint
	@TIMESTAMP=$$(date +"%Y%m%d_%H%M%S"); \
	DEST="$(CKPT_DIR)/mspacman_dqn_backup_$$TIMESTAMP.pth"; \
	cp $(CKPT_DIR)/mspacman_dqn.pth $$DEST; \
	echo "$(GREEN)[BACKUP] Backup cree : $$DEST$(RESET)"

##
# @brief  Liste tous les checkpoints .pth disponibles avec leur taille.
#
.PHONY: list-backups
list-backups:
	@echo "$(CYAN)[BACKUPS] Checkpoints disponibles :$(RESET)"
	@ls -lh $(CKPT_DIR)/*.pth 2>/dev/null || \
	 echo "$(YELLOW)  Aucun checkpoint trouve$(RESET)"

##
# @brief  Installe toutes les dependances Python du projet via pip.
#
.PHONY: install
install:
	@echo "$(GREEN)[INSTALL] Installation des dependances...$(RESET)"
	pip install torch torchvision numpy gymnasium ale-py matplotlib imageio Pillow
	@echo "$(GREEN)[INSTALL] Dependances installees$(RESET)"

##
# @brief  Verifie que l'environnement Python et les libs sont correctement installes.
# @details Affiche versions Python, PyTorch, CUDA, Gymnasium, NumPy.
#
.PHONY: check-env
check-env:
	@echo "$(GREEN)[CHECK] Verification de l'environnement...$(RESET)"
	@$(PYTHON) -c "\
import sys; print(f'  Python     : {sys.version.split()[0]}'); \
import torch; print(f'  PyTorch    : {torch.__version__}'); \
print(f'  CUDA dispo : {torch.cuda.is_available()}'); \
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'; \
print(f'  GPU        : {gpu}'); \
import gymnasium; print(f'  Gymnasium  : {gymnasium.__version__}'); \
import numpy; print(f'  NumPy      : {numpy.__version__}'); \
"
	@echo "$(GREEN)[CHECK] Environnement OK$(RESET)"

# =============================================================================
# NETTOYAGE
# =============================================================================

##
# @brief  Supprime le checkpoint principal et log.json (garde backups et videos).
#
.PHONY: clean
clean:
	@echo "$(YELLOW)[CLEAN] Nettoyage checkpoint et logs...$(RESET)"
	@read -p "Supprimer $(CKPT_DIR)/mspacman_dqn.pth et $(LOG_FILE) ? [y/N] " CONFIRM; \
	if [ "$$CONFIRM" = "y" ] || [ "$$CONFIRM" = "Y" ]; then \
		rm -f $(CKPT_DIR)/mspacman_dqn.pth; \
		rm -f $(LOG_FILE); \
		echo "$(GREEN)[CLEAN] Nettoyage effectue$(RESET)"; \
	else \
		echo "$(CYAN)[CLEAN] Annule$(RESET)"; \
	fi

##
# @brief  Supprime toutes les videos MP4 enregistrees.
#
.PHONY: clean-videos
clean-videos:
	@echo "$(YELLOW)[CLEAN] Suppression des videos...$(RESET)"
	rm -f $(VIDEO_DIR)/*.mp4
	@echo "$(GREEN)[CLEAN] Videos supprimees$(RESET)"

##
# @brief  Supprime la documentation generee et le Doxyfile.
#
.PHONY: clean-doc
clean-doc:
	@echo "$(YELLOW)[CLEAN] Suppression de la documentation...$(RESET)"
	rm -rf $(DOC_DIR) $(DOXYFILE)
	@echo "$(GREEN)[CLEAN] Documentation supprimee$(RESET)"

##
# @brief  Supprime tout : checkpoints, backups, logs, videos, docs.
# @details Demande une confirmation avant suppression.
#
.PHONY: clean-all
clean-all:
	@echo "$(RED)[CLEAN-ALL] Suppression COMPLETE (checkpoints, logs, videos, docs)$(RESET)"
	@read -p "Confirmer ? [y/N] " CONFIRM; \
	if [ "$$CONFIRM" = "y" ] || [ "$$CONFIRM" = "Y" ]; then \
		rm -rf $(CKPT_DIR)/*.pth $(LOG_FILE) $(VIDEO_DIR)/*.mp4 $(DOC_DIR) $(DOXYFILE); \
		echo "$(GREEN)[CLEAN-ALL] Nettoyage complet effectue$(RESET)"; \
	else \
		echo "$(CYAN)[CLEAN-ALL] Annule$(RESET)"; \
	fi

# =============================================================================
# GARDES (verifications prealables internes)
# =============================================================================

## @brief Verifie que le checkpoint existe avant test/record.
.PHONY: _check_checkpoint
_check_checkpoint:
	@if [ ! -f "$(CKPT_DIR)/mspacman_dqn.pth" ]; then \
		echo "$(RED)[ERROR] Aucun checkpoint : $(CKPT_DIR)/mspacman_dqn.pth$(RESET)"; \
		echo "$(YELLOW)        Lancez d'abord : make train$(RESET)"; \
		exit 1; \
	fi

## @brief Verifie que log.json existe avant plot/status.
.PHONY: _check_log
_check_log:
	@if [ ! -f "$(LOG_FILE)" ]; then \
		echo "$(RED)[ERROR] Aucun log : $(LOG_FILE)$(RESET)"; \
		echo "$(YELLOW)        Lancez d'abord : make train$(RESET)"; \
		exit 1; \
	fi

## @brief Verifie que doxygen est installe.
.PHONY: _check_doxygen
_check_doxygen:
	@which doxygen > /dev/null 2>&1 || { \
		echo "$(RED)[ERROR] Doxygen non installe$(RESET)"; \
		echo "$(YELLOW)        Installez avec : sudo apt install doxygen$(RESET)"; \
		exit 1; \
	}

# =============================================================================
# AIDE
# =============================================================================

##
# @brief  Affiche toutes les cibles disponibles avec leur description.
#
.PHONY: help
help:
	@echo ""
	@echo "$(BLUE)+--------------------------------------------------+$(RESET)"
	@echo "$(BLUE)|        DQN MsPacman -- Makefile v2.1             |$(RESET)"
	@echo "$(BLUE)+--------------------------------------------------+$(RESET)"
	@echo ""
	@echo "$(GREEN)  ENTRAINEMENT$(RESET)"
	@echo "    make train          Lance train.py (reprend si checkpoint)"
	@echo ""
	@echo "$(GREEN)  TEST ET VISUALISATION$(RESET)"
	@echo "    make test           Partie visuelle epsilon=0"
	@echo "    make plot           Courbes depuis log.json"
	@echo "    make record         Enregistre parties >= 159 dots en MP4"
	@echo ""
	@echo "$(GREEN)  DOCUMENTATION$(RESET)"
	@echo "    make doc            Genere HTML Doxygen dans docs/"
	@echo "    make doc-open       Genere + ouvre dans le navigateur"
	@echo ""
	@echo "$(GREEN)  UTILITAIRES$(RESET)"
	@echo "    make status         Stats entrainement (best, loss, eps...)"
	@echo "    make backup         Copie checkpoint avec timestamp"
	@echo "    make list-backups   Liste les .pth disponibles"
	@echo "    make check-env      Verifie Python, CUDA, dependances"
	@echo "    make install        pip install toutes les dependances"
	@echo ""
	@echo "$(GREEN)  NETTOYAGE$(RESET)"
	@echo "    make clean          Supprime checkpoint + log.json"
	@echo "    make clean-videos   Supprime les MP4 enregistres"
	@echo "    make clean-doc      Supprime docs/ et Doxyfile"
	@echo "    make clean-all      Supprime TOUT (avec confirmation)"
	@echo ""
	@echo "$(CYAN)  Workflow typique :$(RESET)"
	@echo "    make install        # Installer les dependances"
	@echo "    make check-env      # Verifier CUDA / Python"
	@echo "    make train          # Entrainer l'agent"
	@echo "    make status         # Verifier la progression"
	@echo "    make plot           # Visualiser les courbes"
	@echo "    make test           # Voir l'agent jouer"
	@echo "    make backup         # Sauvegarder le checkpoint"
	@echo "    make doc            # Generer la documentation"
	@echo ""
