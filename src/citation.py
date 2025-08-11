import gzip
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Set, Dict, Any, Optional, Tuple
from uuid import uuid4

import pydot
#from gard import GARD
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import NodeWithScore
from metapub import PubMedFetcher
from pybtex.database import parse_string
from pybtex.plugin import find_plugin

logger = logging.getLogger(__name__)
#gard = GARD()

# Import LightRAG citation formatter
try:
    from lightrag_integration.citation_formatter import LightRAGCitationFormatter, LightRAGCitationResult
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False
    logger.warning("LightRAG citation formatter not available")


def onlineFullCitation(pmid: str, citation: str):
    """
    Search PMC by PMID. Get article title, authors,
    journal and abstract. (max 3 req per second w/out API, 10 with)
    Args:
        pmid(str): PMID to create citation for
        citation(str): original citation str
    Returns:
        citation(str): formatted citation
    """
    full_citation = ""
    fetch = PubMedFetcher()
    try:
        article = fetch.article_by_pmid(pmid)
    except:
        return f"[{citation}](https://pubmed.ncbi.nlm.nih.gov/{pmid})"
    # extract metadata
    full_citation += article.title
    full_citation += f"\nJOURNAL: {article.journal}, {article.year}\nAUTHORS: {', '.join(article.authors)}\n"
    # add link
    full_citation += f"https://pubmed.ncbi.nlm.nih.gov/{pmid}\n"
    return full_citation


def bib_to_apa7_html(bibtex):
    bibliography = parse_string(bibtex, "bibtex")
    formatted_bib = APA.format_bibliography(bibliography)
    return "\n".join(entry.text.render(TEXT) for entry in formatted_bib)


def find_text(element, tag):
    e = element.find(tag)
    if e is not None:
        return e.text
    return ""


APA = find_plugin("pybtex.style.formatting", "apa")()
TEXT = find_plugin("pybtex.backends", "text")()


def pmid_to_bib(pmid):
    pmid = int(pmid)
    padded_pmid = f"{pmid:08d}"
    with gzip.open(
        f"/data/Archive/pubmed/Archive/{padded_pmid[:2]}/{padded_pmid[2:4]}/{padded_pmid[4:6]}/{pmid}.xml.gz"
    ) as f:
        xml = f.read()
    root = ET.fromstring(xml)
    authors = []
    for a in root.findall(".//Author"):
        lastname = find_text(a, ".//LastName")
        forename = find_text(a, ".//ForeName")
        authors.append(f"{lastname}, {forename}")
    author = " and ".join(authors)
    if author == "":
        author = "Anonymous"
    title = find_text(root, ".//ArticleTitle")
    journal = find_text(root, ".//Journal/Title")
    year = find_text(root, ".//PubDate/Year")
    month = find_text(root, ".//PubDate/Month")
    doi = find_text(root, ".//ArticleId[@IdType='doi']")
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    bibtex = f"""
    @article{{{pmid},
            author = {{{author}}},
            title = {{{title}}},
            journal = {{{journal}}},
            year = {{{year}}},
            month = {{{month}}},
            doi = {{{doi}}},
            url = {{{url}}},
        }}
    """
    return bibtex


def generate_full_pmid_citation(pmid):
    bibtex = pmid_to_bib(pmid)
    citation = bib_to_apa7_html(bibtex)
    return citation


def format_citation(citation: str):
    """
    Uses article ID to generate a URL for the source.
    Returns formatted link, with the citation as link text.
    """
    if citation.startswith("PMID:"):
        pmid = citation.removeprefix("PMID:")
        try:
            return generate_full_pmid_citation(pmid)
        except FileNotFoundError:
            logger.exception(f"Could not find file for PMID {pmid}")
            return citation
    elif citation.startswith("ORPHA:"):
        orpha_code = citation.removeprefix("ORPHA:")
        return f"[{citation}](https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=EN&Expert={orpha_code})"
    elif citation.startswith("OMIM:"):
        omim_identifier = citation.removeprefix("OMIM:")
        return f"[{citation}](https://www.omim.org/entry/{omim_identifier})"
    elif citation.startswith("UMLS:"):
        umls_identifier = citation.removeprefix("UMLS:")
        # temp fix for broken UMLS links
        return f"[{citation}](https://www.ncbi.nlm.nih.gov/medgen/?term={umls_identifier})"
        # return f"[{citation}](https://uts.nlm.nih.gov/metathesaurus.html#?searchString={umls_identifier})"
    #elif citation.startswith("GARD:"):
        #gard_identifier = citation.removeprefix("GARD:")
        #return f"[{citation}]({gard.get_url(gard_identifier)})"
    else:
        return citation


def format_citations2(citations: List[str]):
    citations_formatted = []
    for citation in citations:
        citations_formatted.append(format_citation(citation))
    return citations_formatted


async def get_formatted_sources(source_nodes: List[NodeWithScore]):
    """
    Return formatted string of source numbers and corresponding urls.
    Return dict mapping original source numbers to new source numbers.
    """
    references = "\n\n### Sources\n"
    citations_dict = {}
    sources_dict = {}

    for node in source_nodes:
        # get source number
        text = node.text
        source_number = int(text.split(":")[0].removeprefix("Source "))
        # get list of formatted source URLs
        citations = format_citations2(node.metadata["citation"])
        # add URLs and source number to dict
        if len(citations) == 1:
            try:
                citations_dict[citations[0]].append(source_number)
            except KeyError:
                citations_dict[citations[0]] = [source_number]
        else:  # more than one citation for same source number :(
            for citation in citations:
                # add a unique source number for each citation
                try:
                    citations_dict[citation].append(source_number)
                except KeyError:
                    citations_dict[citation] = [source_number]
                source_number += 0.1  # make source number unique

    # consolidate citations w/ multiple source numbers to have just one
    for c in citations_dict:
        # use min source number for citation
        new_source = min(citations_dict[c])
        references += f"[{new_source}] {str(c)}"
        # if more than one source, track replaced numbers
        if len(citations_dict[c]) > 1:
            for source in citations_dict[c]:
                sources_dict[source] = new_source
        else:
            sources_dict[new_source] = new_source
    return references, sources_dict


def get_source_ordering(source_order, source_number: str):
    try:
        return source_order.index(int(source_number))
    except ValueError:
        return len(source_order)


def generate_bibliography(source_nodes: List[NodeWithScore], source_order: List[int]):
    references = "\n\n### Sources\n"
    # deduplicate sources
    source_map = {}
    for node in source_nodes:
        source_number = int(node.text.split(":")[0].removeprefix("Source "))
        candidate_citations = node.metadata["citation"]
        citation = candidate_citations[0]
        print(f"Source {source_number}: {citation}")
        if source_number not in source_map:
            source_map[source_number] = format_citation(citation)

    inline_citation_map = defaultdict(list)
    citation_bibliography_number = {}
    for i, (source_number, citation) in enumerate(
        sorted(source_map.items(), key=lambda x: get_source_ordering(source_order, x[0]))
    ):
        if citation not in citation_bibliography_number:
            bibliography_number = len(citation_bibliography_number) + 1
            citation_bibliography_number[citation] = bibliography_number
            references += f"[{bibliography_number}] {citation}\n"
        else:
            bibliography_number = citation_bibliography_number[citation]
        inline_citation_map[source_number].append(bibliography_number)

    return references, inline_citation_map


def get_source_nodes(response:RESPONSE_TYPE, content:str, sources:Set[int]):
    source_nodes = []
    for source_node in response.source_nodes:
        source = int(source_node.text.split(":")[0].removeprefix("Source "))
        if source in sources:
            source_nodes.append(source_node)
    return source_nodes


def get_sources(content: str):
    sources = set(map(int, re.findall(r"SOURCE (\d+)", content, re.I)))
    matches = re.findall(r"SOURCES ([\d, ]+)", content)
    if matches:
        sources.update(map(int, matches[0].split(",")))
    return sources


def get_source_graph(source_nodes: List[NodeWithScore]):
    graph = pydot.Dot("source_graph", graph_type="digraph")

    for node in source_nodes:
        subj = node.metadata["subject"]
        obj = node.metadata["object"]
        predicate = node.metadata["predicate"]
        graph.add_edge(pydot.Edge(subj, obj, label=predicate))

    filename = f".files/{uuid4()}.png"
    graph.write_png(filename)
    return filename


def smart_inline_citation_format(numbers: List[int]):
    # calculate consecutive ranges
    # join those with -
    # join all with ,
    numbers.sort()
    ranges = []
    start = numbers[0]
    end = numbers[0]
    for number in numbers[1:]:
        if number == end + 1:
            end = number
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = number
            end = number
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    cites = ", ".join(ranges)
    return f"[{cites}]"


def merge_adjacent_citations(content: str):
    # test case
    # GNE Myopathy is a rare genetic disorder characterized by progressive muscle weakness and atrophy, primarily affecting skeletal muscles. The condition is caused by mutations in the GNE gene, which encodes an enzyme involved in the synthesis of sialic acid, a crucial component of cell membranes and various glycoproteins [1-10], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26].\n\nThe symptoms of GNE Myopathy typically manifest during early childhood or adolescence and include muscle weakness, muscle atrophy, and abnormal electrical activity in muscles as detected by electromyography (EMG) tests [2], [3], [4], [5], [6]. Over time, the disease can lead to significant disability and impaired mobility.\n\nThe GNE gene has multiple allelic variants associated with GNE Myopathy, which are responsible for the different forms of the disorder observed in affected individuals [17], [18], [19], [20]. The specific variant determines the severity and progression of the disease, as well as the age of onset and other clinical features.\n\nIn summary, GNE Myopathy is a genetic disorder characterized by muscle weakness, atrophy, and electrical abnormalities in muscles due to mutations in the GNE gene. The condition affects skeletal muscles and can lead to significant disability over time.
    content = re.sub(r"\], \[", ", ", content)
    citations = re.findall(r"(\[[\d,\- ]+\])", content)
    mapping = {}
    for cite in citations:
        numbers = get_numbers_complex(cite)
        mapping[cite] = smart_inline_citation_format(numbers)
    for x, y in mapping.items():
        content = content.replace(x, y)
    return content


def get_numbers_complex(cite):
    x = cite.removeprefix("[").removesuffix("]").split(",")
    numbers = set()
    for a in x:
        if "-" in a:
            start, end = map(int, a.split("-"))
            numbers.update(range(start, end + 1))
        try:
            numbers.add(int(a))
        except ValueError:
            pass
    numbers = list(numbers)
    return numbers


def get_numbers(source):
    numbers = source.split(",")
    numbers = map(str.strip, numbers)
    numbers = filter(None, numbers)
    numbers = [int(x) for x in numbers]
    return numbers


def expand_citations(content: str):
    # (Sources 3, 8, and 9) -> (Source 3, Source 8, Source 9)
    sources = re.findall(r"Sources (\d[\d,\- ]*), and (\d[\d,\- ]*)", content)
    for source, and_source in sources:
        numbers = get_numbers_complex(source)
        numbers.extend(get_numbers_complex(and_source))
        content = content.replace(f"Sources {source}, and {and_source}", ", ".join([f"Source {x}" for x in numbers]))
     # (Sources 3, 8 and 9) -> (Source 3, Source 8, Source 9)
    sources = re.findall(r"Sources (\d[\d,\- ]*) and (\d[\d,\- ]*)", content)
    for source, and_source in sources:
        numbers = get_numbers_complex(source)
        numbers.extend(get_numbers_complex(and_source))
        content = content.replace(f"Sources {source} and {and_source}", ", ".join([f"Source {x}" for x in numbers]))
    # (Sources 9-12) -> (Source 9, Source 10, Source 11, Source 12)
    sources = re.findall(r"Sources* (\d[\d,\- ]*)-([\d[\d,\- ]*)", content)
    for start, end in sources:
        end = get_numbers_complex(end)[-1]
        numbers = list(range(int(start), end + 1))
        if f"Sources {start}-{end}" in content:
            content = content.replace(f"Sources {start}-{end}", ", ".join([f"Source {x}" for x in numbers]))
        else:
            content = content.replace(f"Source {start}-{end}", ", ".join([f"Source {x}" for x in numbers]))
    # (Sources 5, 6, 7) -> (Source 5), (Source 6), (Source 7)
    sources = re.findall(r"Sources (\d[\d,\- ]*)", content)
    for source in sources:
        numbers = get_numbers_complex(source)
        if len(numbers) == 1:
            content = content.replace(f"Sources {source}", f"Source {numbers[0]}")
        else:
            content = content.replace(f"Sources {source}", ", ".join([f"Source {x}" for x in numbers]))
    # [3, 4, and 5] -> [3, 4, 5]
    sources = re.findall(r"\[(\d[\d,\- ]*), and (\d[\d,\- ]*)\]", content)
    for source, and_source in sources:
        numbers = get_numbers_complex(source)
        numbers.append(int(and_source))
        content = content.replace(f"{source}, and {and_source}", ", ".join([f"[{x}]" for x in numbers]))
    # [3, 4, 5] -> [3], [4], [5]
    sources = re.findall(r"\[(\d[\d,\- ]+)\]", content)
    for source in sources:
        numbers = get_numbers_complex(source)
        content = content.replace(source, ", ".join([f"[{x}]" for x in numbers]))
    return content


def postprocess_citation(response):
    content = response.response
    print("LLM Output:")
    print(content)

    content = content.split("Sources:")[0].strip()
    content = content.split("\n\nSource:")[0].strip()
    content = content.split("References:")[0].strip()
    content = expand_citations(content)
    content = normalize_citations(content)


    print("Normalized Citations:")
    print(content)

    source_order = get_source_order(content)
    print('Source Order:', source_order)

    source_nodes =  get_source_nodes(response, content, source_order)
    print("Source Nodes: ", len(source_nodes))

    # remove extraneous hallucinated sources
    # if source_order:
    #     for i in range(len(source_nodes) + 1, max(source_order) + 1):
    #         content = re.sub(rf"\W*\[{i}\],", "", content)
    #         content = re.sub(rf"\W*\[{i}\]", "", content)

    bibliography = None
    if source_nodes:
        source_order = get_source_order(content)
        print('Source Order:', source_order)
        bibliography, inline_citation_map = generate_bibliography(source_nodes, source_order)
        for source_number, biblography_numbers in inline_citation_map.items():
            content = re.sub(rf"\[{source_number}\]", smart_inline_citation_format(biblography_numbers), content)
            content = merge_adjacent_citations(content)

    print('With Citations:')
    print(content)
    print('Bibliography:')
    print(bibliography)
    return content, bibliography


def normalize_citations(content):
    content = re.sub(r"Source (\d+)", r"[\1]", content, flags=re.I)
    content = re.sub(r"\(([\d, -]+)\)[., ]", r"[\1].", content)
    content = re.sub(r"\(\[", "[", content)
    content = re.sub(r"\]\)", "]", content)
    content = re.sub(r"\[\[+", "[", content)
    content = re.sub(r"\]\]+", "]", content)
    return content


def get_source_order(content):
    source_order = re.findall(r"\[(\d+)\]", content)
    source_order = [int(source) for source in source_order]
    return source_order


def format_lightrag_citations(
    content: str,
    source_documents: List[str],
    entities_used: List[Dict[str, Any]],
    relationships_used: List[Dict[str, Any]],
    confidence_score: float,
    config=None
) -> Tuple[str, str]:
    """
    Format citations for LightRAG responses with PDF document sources.
    
    Args:
        content: The response content to add citations to
        source_documents: List of source document file paths
        entities_used: List of entities used in the response
        relationships_used: List of relationships used in the response
        confidence_score: Overall response confidence score
        config: Optional configuration object
    
    Returns:
        Tuple of (formatted_content, bibliography)
    """
    if not LIGHTRAG_AVAILABLE:
        logger.warning("LightRAG citation formatter not available, returning original content")
        return content, ""
    
    try:
        formatter = LightRAGCitationFormatter(config)
        result = formatter.format_lightrag_citations(
            content, source_documents, entities_used, relationships_used, confidence_score
        )
        
        logger.info(f"Formatted LightRAG citations: {result.source_count} sources")
        return result.formatted_content, result.bibliography
        
    except Exception as e:
        logger.error(f"Error formatting LightRAG citations: {str(e)}", exc_info=True)
        return content, f"Error generating bibliography: {str(e)}"


def process_lightrag_response(response_data: Dict[str, Any], config=None) -> Tuple[str, str]:
    """
    Process a complete LightRAG response and format citations.
    
    Args:
        response_data: Dictionary containing LightRAG response data with keys:
            - answer: The response text
            - source_documents: List of source document paths
            - entities_used: List of entities used
            - relationships_used: List of relationships used
            - confidence_score: Overall confidence score
        config: Optional configuration object
    
    Returns:
        Tuple of (formatted_content, bibliography)
    """
    try:
        content = response_data.get("answer", "")
        source_documents = response_data.get("source_documents", [])
        entities_used = response_data.get("entities_used", [])
        relationships_used = response_data.get("relationships_used", [])
        confidence_score = response_data.get("confidence_score", 0.0)
        
        return format_lightrag_citations(
            content, source_documents, entities_used, 
            relationships_used, confidence_score, config
        )
        
    except Exception as e:
        logger.error(f"Error processing LightRAG response: {str(e)}", exc_info=True)
        return response_data.get("answer", ""), f"Error processing citations: {str(e)}"


def merge_citation_systems(
    lightrag_content: str,
    lightrag_bibliography: str,
    traditional_content: str,
    traditional_bibliography: str
) -> Tuple[str, str]:
    """
    Merge citations from LightRAG and traditional citation systems.
    
    This function is useful for hybrid responses that combine information
    from both LightRAG (PDF sources) and other systems (e.g., Perplexity API).
    
    Args:
        lightrag_content: Content with LightRAG citations
        lightrag_bibliography: LightRAG bibliography
        traditional_content: Content with traditional citations
        traditional_bibliography: Traditional bibliography
    
    Returns:
        Tuple of (merged_content, merged_bibliography)
    """
    try:
        # Combine content
        if lightrag_content and traditional_content:
            merged_content = f"{traditional_content}\n\n**Additional Information from Knowledge Base:**\n{lightrag_content}"
        elif lightrag_content:
            merged_content = lightrag_content
        else:
            merged_content = traditional_content
        
        # Combine bibliographies
        merged_bibliography = ""
        if traditional_bibliography:
            merged_bibliography += traditional_bibliography
        
        if lightrag_bibliography:
            if merged_bibliography:
                merged_bibliography += f"\n\n{lightrag_bibliography}"
            else:
                merged_bibliography = lightrag_bibliography
        
        return merged_content, merged_bibliography
        
    except Exception as e:
        logger.error(f"Error merging citation systems: {str(e)}", exc_info=True)
        # Return the best available content on error
        content = lightrag_content or traditional_content or ""
        bibliography = lightrag_bibliography or traditional_bibliography or ""
        return content, bibliography


def get_lightrag_citation_statistics(
    source_documents: List[str],
    entities_used: List[Dict[str, Any]],
    relationships_used: List[Dict[str, Any]],
    config=None
) -> Dict[str, Any]:
    """
    Get statistics about LightRAG citations.
    
    Args:
        source_documents: List of source document paths
        entities_used: List of entities used
        relationships_used: List of relationships used
        config: Optional configuration object
    
    Returns:
        Dictionary containing citation statistics
    """
    if not LIGHTRAG_AVAILABLE:
        return {"error": "LightRAG citation formatter not available"}
    
    try:
        formatter = LightRAGCitationFormatter(config)
        
        # Extract citations to get statistics
        citations = formatter._extract_pdf_citations(
            source_documents, entities_used, relationships_used
        )
        
        return formatter.get_citation_statistics(citations)
        
    except Exception as e:
        logger.error(f"Error getting LightRAG citation statistics: {str(e)}", exc_info=True)
        return {"error": str(e)}


def calculate_lightrag_confidence(
    base_confidence: float,
    source_documents: List[str],
    entities_used: List[Dict[str, Any]],
    relationships_used: List[Dict[str, Any]],
    query_context: Optional[Dict[str, Any]] = None,
    config=None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate enhanced confidence score for LightRAG responses.
    
    Args:
        base_confidence: Base confidence from query processing
        source_documents: List of source document paths
        entities_used: List of entities used in the response
        relationships_used: List of relationships used in the response
        query_context: Optional context about the query
        config: Optional configuration object
    
    Returns:
        Tuple of (enhanced_confidence, confidence_breakdown_dict)
    """
    if not LIGHTRAG_AVAILABLE:
        logger.warning("LightRAG confidence scoring not available")
        return base_confidence, {"error": "Confidence scoring not available"}
    
    try:
        from lightrag_integration.confidence_scoring import LightRAGConfidenceScorer
        
        scorer = LightRAGConfidenceScorer(config)
        breakdown = scorer.calculate_response_confidence(
            base_confidence, source_documents, entities_used, relationships_used, query_context
        )
        
        # Convert breakdown to dictionary for serialization
        breakdown_dict = {
            "overall_confidence": breakdown.overall_confidence,
            "base_confidence": breakdown.base_confidence,
            "enhancement_factor": breakdown.enhancement_factor,
            "explanation": breakdown.explanation,
            "source_scores": breakdown.source_scores,
            "entity_scores": breakdown.entity_scores,
            "relationship_scores": breakdown.relationship_scores,
            "confidence_factors": {
                "entity_confidence": breakdown.confidence_factors.entity_confidence,
                "relationship_confidence": breakdown.confidence_factors.relationship_confidence,
                "source_reliability": breakdown.confidence_factors.source_reliability,
                "graph_connectivity": breakdown.confidence_factors.graph_connectivity,
                "evidence_consistency": breakdown.confidence_factors.evidence_consistency,
                "citation_quality": breakdown.confidence_factors.citation_quality,
                "temporal_relevance": breakdown.confidence_factors.temporal_relevance
            }
        }
        
        logger.info(f"Enhanced confidence: {breakdown.overall_confidence:.3f} (base: {base_confidence:.3f})")
        return breakdown.overall_confidence, breakdown_dict
        
    except Exception as e:
        logger.error(f"Error calculating LightRAG confidence: {str(e)}", exc_info=True)
        return base_confidence, {"error": str(e)}


def get_lightrag_confidence_display_info(
    confidence_score: float,
    confidence_breakdown: Optional[Dict[str, Any]] = None,
    config=None
) -> Dict[str, Any]:
    """
    Get display information for LightRAG confidence scores.
    
    Args:
        confidence_score: The confidence score to display
        confidence_breakdown: Optional detailed confidence breakdown
        config: Optional configuration object
    
    Returns:
        Dictionary with display information (color, icon, text, tooltip)
    """
    if not LIGHTRAG_AVAILABLE:
        return {
            "confidence_score": confidence_score,
            "confidence_level": "Unknown",
            "color": "gray",
            "icon": "?",
            "display_text": f"Confidence: {confidence_score:.1%}",
            "tooltip": "Confidence scoring not available",
            "show_warning": True
        }
    
    try:
        from lightrag_integration.confidence_scoring import LightRAGConfidenceScorer, ConfidenceBreakdown, ConfidenceFactors
        
        scorer = LightRAGConfidenceScorer(config)
        
        # Create a minimal breakdown if not provided
        if confidence_breakdown:
            breakdown = ConfidenceBreakdown(
                overall_confidence=confidence_breakdown.get("overall_confidence", confidence_score),
                base_confidence=confidence_breakdown.get("base_confidence", confidence_score),
                enhancement_factor=confidence_breakdown.get("enhancement_factor", 0.0),
                confidence_factors=ConfidenceFactors(**confidence_breakdown.get("confidence_factors", {})),
                source_scores=confidence_breakdown.get("source_scores", {}),
                entity_scores=confidence_breakdown.get("entity_scores", {}),
                relationship_scores=confidence_breakdown.get("relationship_scores", {}),
                explanation=confidence_breakdown.get("explanation", "")
            )
        else:
            breakdown = ConfidenceBreakdown(
                overall_confidence=confidence_score,
                base_confidence=confidence_score,
                enhancement_factor=0.0,
                confidence_factors=ConfidenceFactors(),
                source_scores={},
                entity_scores={},
                relationship_scores={},
                explanation=f"Confidence: {confidence_score:.2f}"
            )
        
        return scorer.get_confidence_display_info(breakdown)
        
    except Exception as e:
        logger.error(f"Error getting confidence display info: {str(e)}", exc_info=True)
        return {
            "confidence_score": confidence_score,
            "confidence_level": "Error",
            "color": "red",
            "icon": "!",
            "display_text": f"Confidence: {confidence_score:.1%}",
            "tooltip": f"Error: {str(e)}",
            "show_warning": True
        }
