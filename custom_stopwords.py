"""
Turkish Mathematics Curriculum NLP Analysis - Custom Stopwords
=============================================================
This module provides custom stopwords for Turkish curriculum analysis.
It can be imported by different scripts in the project to ensure consistency.
"""

def get_turkish_curriculum_stopwords():
    """
    Returns a set of custom stopwords specifically for Turkish mathematics curriculum analysis.
    These words are common in educational contexts but don't contribute significantly to topic modeling.
    """
    # Basic Turkish stopwords (if NLTK's Turkish stopwords aren't available)
    basic_turkish_stopwords = {
        've', 'ile', 'bu', 'bir', 'için', 'ya', 'de', 'da', 'olarak', 'gibi',
        'kadar', 'değil', 'daha', 'çok', 'en', 'göre', 'her', 'mi', 'ne',
        'o', 'ama', 'ki', 'eğer', 'veya', 'hem', 'ise', 'ancak', 'şey'
    }
    
    # Numbers and basic math symbols
    numbers_and_symbols = {
        'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on',
        'a', 'b', 'c', 'ç', 'd', 'e', 'f', 'x', 'y', 'z', 'n'
    }
    
    
    # Combine all sets
    all_stopwords = basic_turkish_stopwords.union(numbers_and_symbols)
    
    return all_stopwords

def get_extended_stopwords():
    """
    Returns the extended set of stopwords, including the Turkish curriculum 
    stopwords plus additional words for more aggressive filtering.
    """
    base_stopwords = get_turkish_curriculum_stopwords()
    
    # Additional stopwords for more aggressive filtering
    additional_stopwords = {
        'kullanarak', 'yapılır', 'gerekli', 'uygulanır', 'oluşan',
        'oluştururlar', 'oluşturulan', 'oluşturan', 'sahip', 'sahiptir',
        'olmak', 'olur', 'olarak', 'olanlar', 'olmalıdır',
        'farklı', 'aynı', 'benzer', 'diğer', 'başka',
        'çeşitli', 'tüm', 'her', 'bazı', 'hiç',
        'konu', 'konular', 'özellik', 'özellikler', 'örnek', 'örnekler'
    }
    
    return base_stopwords.union(additional_stopwords)