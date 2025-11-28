#!/usr/bin/env python3
"""
Script para verificar y analizar las referencias bibliográficas de Aurora-GLM
Autor: Lucy E. Arias
Fecha: 2025-11-26
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Reference:
    """Representa una referencia bibliográfica"""
    authors: str
    year: str
    title: str
    publication: str
    ref_type: str  # 'book', 'article', 'software'
    location: str  # Donde se encuentra en el código
    
    def __str__(self):
        return f"{self.authors} ({self.year}). {self.title}"

def extract_references_from_md(filepath: str) -> List[Reference]:
    """Extrae referencias del archivo REFERENCES.md"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    references = []
    
    # Patrón para libros: Autor(es), Año. Título. Editorial
    book_pattern = r'([A-Z][^.]+?),\s+([A-Z][^.]+?)\s+\((\d{4})\)\.\s+\*([^*]+)\*\.?\s*([^.\n]+)?'
    for match in re.finditer(book_pattern, content):
        authors, rest, year, title, publisher = match.groups()
        if len(authors) < 50:  # Filtrar coincidencias falsas
            references.append(Reference(
                authors=authors.strip(),
                year=year,
                title=title.strip(),
                publication=publisher.strip() if publisher else "Unknown",
                ref_type='book',
                location='REFERENCES.md'
            ))
    
    # Patrón para artículos: Autor(es). (Año). "Título". Revista, Vol(Num), páginas
    article_pattern = r'([A-Z][^.]+?)\.\s+\((\d{4})\)\.\s+"([^"]+)"\.\s+\*([^*]+)\*,?\s*(\d+)?\(?(\d+)?\)?'
    for match in re.finditer(article_pattern, content):
        authors, year, title, journal, volume, issue = match.groups()
        if len(authors) < 100:  # Filtrar coincidencias falsas
            pub_info = f"{journal}"
            if volume:
                pub_info += f", {volume}"
            if issue:
                pub_info += f"({issue})"
            
            references.append(Reference(
                authors=authors.strip(),
                year=year,
                title=title.strip(),
                publication=pub_info.strip(),
                ref_type='article',
                location='REFERENCES.md'
            ))
    
    return references

def extract_dois_from_code(root_dir: str) -> List[Tuple[str, str]]:
    """Extrae enlaces DOI del código fuente"""
    dois = []
    root = Path(root_dir)
    
    # Buscar en archivos Python
    for py_file in root.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Buscar DOIs
            doi_pattern = r'https://doi\.org/([\w./\-]+)'
            for match in re.finditer(doi_pattern, content):
                doi = match.group(0)
                dois.append((str(py_file.relative_to(root)), doi))
        except Exception as e:
            pass
    
    return dois

def categorize_references(references: List[Reference]) -> Dict[str, List[Reference]]:
    """Categoriza referencias por tipo"""
    categories = {
        'Libros': [],
        'Artículos Científicos': [],
        'Software/Paquetes': []
    }
    
    for ref in references:
        if ref.ref_type == 'book':
            categories['Libros'].append(ref)
        elif ref.ref_type == 'article':
            categories['Artículos Científicos'].append(ref)
        else:
            categories['Software/Paquetes'].append(ref)
    
    return categories

def main():
    print("=" * 100)
    print("VERIFICACIÓN DE REFERENCIAS BIBLIOGRÁFICAS - AURORA-GLM")
    print("=" * 100)
    print()
    
    # Extraer referencias del REFERENCES.md
    print("📚 Extrayendo referencias de REFERENCES.md...")
    references = extract_references_from_md('REFERENCES.md')
    print(f"   ✓ {len(references)} referencias encontradas")
    print()
    
    # Categorizar
    categories = categorize_references(references)
    
    # Mostrar referencias por categoría
    for category, refs in categories.items():
        if refs:
            print(f"\n{'=' * 100}")
            print(f"{category.upper()} ({len(refs)})")
            print('=' * 100)
            
            for i, ref in enumerate(refs, 1):
                print(f"\n{i}. {ref.authors} ({ref.year})")
                print(f"   Título: {ref.title}")
                print(f"   Publicación: {ref.publication}")
    
    # Extraer DOIs del código
    print(f"\n{'=' * 100}")
    print("ENLACES DOI EN EL CÓDIGO FUENTE")
    print('=' * 100)
    
    dois = extract_dois_from_code('.')
    unique_dois = list(set([doi for _, doi in dois]))
    
    print(f"\n✓ {len(unique_dois)} DOIs únicos encontrados:\n")
    for i, doi in enumerate(sorted(unique_dois), 1):
        print(f"{i:2d}. {doi}")
        # Mostrar en qué archivos aparece
        files = [f for f, d in dois if d == doi]
        if len(files) <= 3:
            for f in files:
                print(f"      → {f}")
    
    # Resumen general
    print(f"\n{'=' * 100}")
    print("RESUMEN GENERAL")
    print('=' * 100)
    print(f"📖 Total de libros citados:           {len(categories['Libros'])}")
    print(f"📄 Total de artículos científicos:    {len(categories['Artículos Científicos'])}")
    print(f"💻 Total de software/paquetes:        {len(categories['Software/Paquetes'])}")
    print(f"🔗 Total de DOIs en el código:        {len(unique_dois)}")
    print(f"📊 Total de referencias únicas:       {len(references) + len(unique_dois)}")
    print()
    
    # Verificar consistencia
    print(f"{'=' * 100}")
    print("ANÁLISIS DE CALIDAD")
    print('=' * 100)
    print()
    
    # Referencias con información completa
    complete_refs = [r for r in references if r.publication != "Unknown"]
    print(f"✓ Referencias con información completa: {len(complete_refs)}/{len(references)} "
          f"({len(complete_refs)/len(references)*100:.1f}%)")
    
    # Referencias recientes (últimos 10 años)
    recent_refs = [r for r in references if int(r.year) >= 2015]
    print(f"✓ Referencias recientes (2015+):        {len(recent_refs)}/{len(references)} "
          f"({len(recent_refs)/len(references)*100:.1f}%)")
    
    # Referencias clásicas (fundacionales)
    classic_refs = [r for r in references if int(r.year) < 2000]
    print(f"✓ Referencias clásicas (pre-2000):      {len(classic_refs)}/{len(references)} "
          f"({len(classic_refs)/len(references)*100:.1f}%)")
    
    print()
    print("Verificación completada exitosamente.")
    print()

if __name__ == '__main__':
    main()
