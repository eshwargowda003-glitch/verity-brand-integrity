import os
import re
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    create_engine,
    String,
    Text,
    Integer,
    ForeignKey,
    DateTime,
    select,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
    joinedload,
)

URL = os.getenv('DATABASE_URL', 'sqlite:///./verity.db')
engine = create_engine(
    URL,
    connect_args={'check_same_thread': False} if URL.startswith('sqlite') else {}
)
SessionLocal = sessionmaker(engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Brand(Base):
    __tablename__ = 'brands'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(180), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default='')
    rules: Mapped[list['Rule']] = relationship(
        back_populates='brand', cascade='all, delete-orphan'
    )
    reviews: Mapped[list['Review']] = relationship(
        back_populates='brand', cascade='all, delete-orphan'
    )


class Rule(Base):
    __tablename__ = 'rules'
    id: Mapped[int] = mapped_column(primary_key=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey('brands.id'), index=True)
    kind: Mapped[str] = mapped_column(String(30))
    title: Mapped[str] = mapped_column(String(220))
    pattern: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(20), default='medium')
    explanation: Mapped[str] = mapped_column(Text)
    suggestion: Mapped[str] = mapped_column(Text, default='')
    source_note: Mapped[str] = mapped_column(Text, default='')
    brand: Mapped['Brand'] = relationship(back_populates='rules')


class Review(Base):
    __tablename__ = 'reviews'
    id: Mapped[int] = mapped_column(primary_key=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey('brands.id'))
    title: Mapped[str] = mapped_column(String(220))
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), default='needs_review')
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    brand: Mapped['Brand'] = relationship(back_populates='reviews')
    findings: Mapped[list['Finding']] = relationship(
        back_populates='review',
        cascade='all, delete-orphan',
        order_by='Finding.id',
    )
    decisions: Mapped[list['Decision']] = relationship(
        back_populates='review',
        cascade='all, delete-orphan',
        order_by='Decision.id',
    )


class Finding(Base):
    __tablename__ = 'findings'
    id: Mapped[int] = mapped_column(primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey('reviews.id'), index=True)
    rule_id: Mapped[int | None] = mapped_column(ForeignKey('rules.id'), nullable=True)
    rule_title: Mapped[str] = mapped_column(String(220))
    category: Mapped[str] = mapped_column(String(30))
    severity: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(String(220))
    description: Mapped[str] = mapped_column(Text)
    matched_text: Mapped[str] = mapped_column(Text, default='')
    evidence_status: Mapped[str] = mapped_column(String(30), default='unverified')
    evidence_reference: Mapped[str] = mapped_column(Text, default='')
    suggested_correction: Mapped[str] = mapped_column(Text, default='')
    decision_status: Mapped[str] = mapped_column(String(30), default='pending')
    decision_reason: Mapped[str] = mapped_column(Text, default='', nullable=True)
    review: Mapped['Review'] = relationship(back_populates='findings')


class Decision(Base):
    __tablename__ = 'decisions'
    id: Mapped[int] = mapped_column(primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey('reviews.id'))
    finding_id: Mapped[int] = mapped_column(ForeignKey('findings.id'))
    decision: Mapped[str] = mapped_column(String(30))
    reason: Mapped[str] = mapped_column(Text, default='', nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    review: Mapped['Review'] = relationship(back_populates='decisions')


def init_db(target_engine=engine):
    Base.metadata.create_all(target_engine)
    try:
        with target_engine.connect() as conn:
            cols_findings = [
                row[1]
                for row in conn.exec_driver_sql(
                    'PRAGMA table_info(findings)'
                ).fetchall()
            ]
            if cols_findings and 'decision_reason' not in cols_findings:
                conn.exec_driver_sql(
                    "ALTER TABLE findings ADD COLUMN decision_reason TEXT DEFAULT ''"
                )
            cols_decisions = [
                row[1]
                for row in conn.exec_driver_sql(
                    'PRAGMA table_info(decisions)'
                ).fetchall()
            ]
            if cols_decisions and 'reason' not in cols_decisions:
                conn.exec_driver_sql(
                    "ALTER TABLE decisions ADD COLUMN reason TEXT DEFAULT ''"
                )
            conn.commit()
    except Exception:
        pass


init_db(engine)


VALID_RULE_KINDS = {'prohibited_phrase', 'required_term', 'fact_check'}
VALID_SEVERITIES = {'low', 'medium', 'high', 'critical'}
VALID_DECISIONS = {'accepted', 'rejected', 'dismissed', 'resolved'}


class BrandIn(BaseModel):
    name: str = Field(min_length=2, max_length=180)
    description: str = Field(default='', max_length=2000)

    @field_validator('name')
    @classmethod
    def clean_name(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 2:
            raise ValueError('Brand name must be at least 2 characters.')
        return s

    @field_validator('description')
    @classmethod
    def clean_desc(cls, v: str) -> str:
        return v.strip() if v else ''


class RuleIn(BaseModel):
    kind: str
    title: str = Field(min_length=2, max_length=220)
    pattern: str = Field(min_length=1, max_length=1000)
    severity: str = 'medium'
    explanation: str = Field(min_length=3, max_length=4000)
    suggestion: str = ''
    source_note: str = ''

    @field_validator('kind')
    @classmethod
    def validate_kind(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in VALID_RULE_KINDS:
            raise ValueError(
                f"Unsupported rule kind '{v}'. Supported: {', '.join(sorted(VALID_RULE_KINDS))}"
            )
        return s

    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{v}'. Supported: {', '.join(sorted(VALID_SEVERITIES))}"
            )
        return s

    @field_validator('title')
    @classmethod
    def clean_title(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 2:
            raise ValueError('Rule title must be at least 2 characters.')
        return s

    @field_validator('pattern')
    @classmethod
    def clean_pattern(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError('Rule pattern cannot be empty or whitespace only.')
        return s

    @field_validator('explanation')
    @classmethod
    def clean_explanation(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 3:
            raise ValueError('Rule explanation must be at least 3 characters.')
        return s

    @field_validator('suggestion', 'source_note')
    @classmethod
    def clean_optional(cls, v: str) -> str:
        return v.strip() if v else ''


class ReviewIn(BaseModel):
    brand_id: int
    title: str = 'Untitled review'
    content: str = Field(min_length=3, max_length=20000)

    @field_validator('title')
    @classmethod
    def clean_title(cls, v: str) -> str:
        s = v.strip()
        return s if s else 'Untitled review'

    @field_validator('content')
    @classmethod
    def clean_content(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 3:
            raise ValueError('Content must be at least 3 characters.')
        return s


class DecisionIn(BaseModel):
    finding_id: int
    decision: str
    reason: str = Field(default='', max_length=2000)

    @field_validator('decision')
    @classmethod
    def validate_decision(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in VALID_DECISIONS:
            raise ValueError(
                f"Unsupported decision '{v}'. Supported: 'accepted', 'rejected', 'resolved'."
            )
        if s == 'dismissed':
            return 'rejected'
        return s

    @field_validator('reason')
    @classmethod
    def clean_reason(cls, v: str) -> str:
        return v.strip() if v else ''


app = FastAPI(title='VERITY Brand Integrity API', version='1.1.0')
origins = [
    x.strip()
    for x in os.getenv(
        'CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5500,null,*'
    ).split(',')
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['*'],
)


def db_dep():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def compile_phrase_pattern(pattern: str) -> re.Pattern | None:
    cleaned = pattern.strip()
    if not cleaned:
        return None
    tokens = [re.escape(tok) for tok in cleaned.split()]
    if not tokens:
        return None
    body = r'\s+'.join(tokens)
    prefix = r'(?<!\w)' if re.match(r'^\w', cleaned) else ''
    suffix = r'(?!\w)' if re.search(r'\w$', cleaned) else ''
    return re.compile(f'{prefix}{body}{suffix}', re.IGNORECASE)


def run_engine(content: str, rules: list[Rule]) -> list[dict]:
    out = []
    if not content or not rules:
        return out
    for rule in rules:
        pattern_re = compile_phrase_pattern(rule.pattern)
        if not pattern_re:
            continue

        if rule.kind == 'prohibited_phrase':
            for m in pattern_re.finditer(content):
                out.append(
                    dict(
                        rule_id=rule.id,
                        rule_title=rule.title,
                        category=rule.kind,
                        severity=rule.severity,
                        title=rule.title,
                        description=rule.explanation,
                        matched_text=m.group(),
                        evidence_status='rule_matched',
                        evidence_reference=rule.source_note
                        or f'Configured rule #{rule.id}',
                        suggested_correction=rule.suggestion,
                        decision_status='pending',
                        decision_reason='',
                    )
                )
        elif rule.kind == 'required_term':
            if not pattern_re.search(content):
                out.append(
                    dict(
                        rule_id=rule.id,
                        rule_title=rule.title,
                        category=rule.kind,
                        severity=rule.severity,
                        title=rule.title,
                        description=rule.explanation,
                        matched_text='',
                        evidence_status='unverified',
                        evidence_reference=rule.source_note
                        or f'Configured rule #{rule.id}',
                        suggested_correction=rule.suggestion
                        or f'Include approved term: {rule.pattern.strip()}',
                        decision_status='pending',
                        decision_reason='',
                    )
                )
        elif rule.kind == 'fact_check':
            if not pattern_re.search(content):
                out.append(
                    dict(
                        rule_id=rule.id,
                        rule_title=rule.title,
                        category=rule.kind,
                        severity=rule.severity,
                        title=rule.title,
                        description=rule.explanation,
                        matched_text='',
                        evidence_status='unverified',
                        evidence_reference=rule.source_note
                        or f'Approved fact record #{rule.id}',
                        suggested_correction=rule.suggestion
                        or f'Check against approved value: {rule.pattern.strip()}',
                        decision_status='pending',
                        decision_reason='',
                    )
                )
    return out


def review_json(r: Review) -> dict:
    return {
        'id': r.id,
        'brand_id': r.brand_id,
        'title': r.title,
        'content': r.content,
        'status': r.status,
        'created_at': (
            r.created_at.isoformat()
            if hasattr(r.created_at, 'isoformat')
            else str(r.created_at)
        ),
        'findings': [
            {
                'id': f.id,
                'rule_id': f.rule_id,
                'rule_title': f.rule_title,
                'category': f.category,
                'severity': f.severity,
                'title': f.title,
                'description': f.description,
                'matched_text': f.matched_text,
                'evidence_status': f.evidence_status,
                'evidence_reference': f.evidence_reference,
                'suggested_correction': f.suggested_correction,
                'decision_status': f.decision_status,
                'decision_reason': getattr(f, 'decision_reason', '') or '',
            }
            for f in r.findings
        ],
    }


def calc_status(fs: list[Finding], current_status: str = '') -> str:
    has_accepted = any(f.decision_status == 'accepted' for f in fs)
    has_pending_high_crit = any(
        f.decision_status == 'pending' and f.severity in ('high', 'critical')
        for f in fs
    )

    if has_accepted or has_pending_high_crit:
        return 'changes_required'

    has_pending = any(f.decision_status == 'pending' for f in fs)
    if has_pending:
        return 'needs_review'

    # All findings have been either resolved or rejected
    if current_status == 'approved':
        return 'approved'

    return 'ready_for_approval'


def generate_compliance_report(r: Review, brand: Brand | None) -> dict:
    brand_name = brand.name if brand else f'Brand #{r.brand_id}'
    brand_desc = brand.description if brand else ''

    report_findings = []
    outstanding = []
    resolved_count = 0
    rejected_count = 0
    accepted_count = 0
    pending_count = 0

    for f in r.findings:
        if f.decision_status == 'pending':
            pending_count += 1
            outstanding.append(f.id)
        elif f.decision_status == 'accepted':
            accepted_count += 1
            outstanding.append(f.id)
        elif f.decision_status == 'resolved':
            resolved_count += 1
        elif f.decision_status in ('rejected', 'dismissed'):
            rejected_count += 1

        has_grounded = bool(f.suggested_correction and f.suggested_correction.strip())
        correction_text = (
            f.suggested_correction.strip()
            if has_grounded
            else 'No configured correction available. A human reviewer must provide or approve appropriate phrasing.'
        )

        report_findings.append({
            'id': f.id,
            'rule_id': f.rule_id,
            'rule_name': f.rule_title,
            'rule_type': f.category,
            'severity': f.severity,
            'matched_text': f.matched_text,
            'explanation': f.description,
            'suggested_correction': correction_text,
            'is_correction_grounded': has_grounded,
            'finding_status': f.decision_status,
            'evidence_status': f.evidence_status,
            'rule_match_status': (
                'matched' if f.category == 'prohibited_phrase' else 'missing_from_content'
            ),
            'verification_status': (
                'verified_resolved' if f.decision_status == 'resolved'
                else (
                    'dismissed' if f.decision_status in ('rejected', 'dismissed')
                    else (
                        'policy_violation_confirmed' if f.decision_status == 'accepted'
                        else 'unverified_external'
                    )
                )
            ),
            'evidence_reference': f.evidence_reference,
            'reviewer_decision': (
                f.decision_status if f.decision_status != 'pending' else None
            ),
            'reviewer_reason': getattr(f, 'decision_reason', '') or None,
        })

    unresolved_count = pending_count + accepted_count

    return {
        'report_type': 'VERITY Brand Governance & Compliance Report',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'disclaimer': (
            'This report documents deterministic checks against user-entered'
            ' brand rules and human reviewer decisions. The system checks'
            ' rule compliance and does not independently verify the factual'
            ' truth or falsity of external claims.'
        ),
        'brand': {
            'id': r.brand_id,
            'name': brand_name,
            'description': brand_desc,
        },
        'review': {
            'id': r.id,
            'title': r.title,
            'submitted_content': r.content,
            'created_at': (
                r.created_at.isoformat()
                if hasattr(r.created_at, 'isoformat')
                else str(r.created_at)
            ),
            'overall_status': r.status,
            'summary': {
                'total_findings': len(r.findings),
                'pending_findings': pending_count,
                'accepted_violations': accepted_count,
                'resolved_findings': resolved_count,
                'rejected_findings': rejected_count,
                'outstanding_unresolved_count': unresolved_count,
                'is_ready_for_approval': r.status == 'ready_for_approval',
                'is_approved': r.status == 'approved',
            },
        },
        'findings': report_findings,
        'outstanding_unresolved_findings': [
            f for f in report_findings if f['id'] in outstanding
        ],
    }


@app.get('/')
def root():
    return {
        'name': 'VERITY Brand Integrity API',
        'version': '1.1.0',
        'status': 'online',
        'docs_url': '/docs',
        'health_url': '/health'
    }


@app.get('/health')
def health(db: Session = Depends(db_dep)):
    db.execute(select(1))
    return {'status': 'ok', 'database': 'connected'}


@app.get('/api/brands')
def brands(db: Session = Depends(db_dep)):
    return [
        {'id': b.id, 'name': b.name, 'description': b.description}
        for b in db.scalars(select(Brand).order_by(Brand.name)).all()
    ]


@app.post('/api/brands', status_code=201)
def create_brand(d: BrandIn, db: Session = Depends(db_dep)):
    existing = db.scalar(
        select(Brand).where(func.lower(func.trim(Brand.name)) == d.name.lower())
    )
    if existing:
        raise HTTPException(409, f"Brand '{d.name}' already exists.")
    b = Brand(name=d.name, description=d.description)
    db.add(b)
    db.commit()
    db.refresh(b)
    return {'id': b.id, 'name': b.name, 'description': b.description}


@app.get('/api/brands/{bid}/rules')
def list_rules(bid: int, db: Session = Depends(db_dep)):
    if not db.get(Brand, bid):
        raise HTTPException(404, 'Brand not found.')
    return [
        {
            'id': r.id,
            'brand_id': r.brand_id,
            'kind': r.kind,
            'title': r.title,
            'pattern': r.pattern,
            'severity': r.severity,
            'explanation': r.explanation,
            'suggestion': r.suggestion,
            'source_note': r.source_note,
            'active': True,
        }
        for r in db.scalars(select(Rule).where(Rule.brand_id == bid)).all()
    ]


@app.post('/api/brands/{bid}/rules', status_code=201)
def add_rule(bid: int, d: RuleIn, db: Session = Depends(db_dep)):
    if not db.get(Brand, bid):
        raise HTTPException(404, 'Brand not found.')

    existing = db.scalar(
        select(Rule).where(
            Rule.brand_id == bid,
            Rule.kind == d.kind,
            func.lower(func.trim(Rule.pattern)) == d.pattern.lower(),
        )
    )
    if existing:
        raise HTTPException(
            409,
            f"Rule already exists: brand already has a '{d.kind}' rule with"
            f" pattern '{d.pattern}'.",
        )

    r = Rule(
        brand_id=bid,
        kind=d.kind,
        title=d.title,
        pattern=d.pattern,
        severity=d.severity,
        explanation=d.explanation,
        suggestion=d.suggestion,
        source_note=d.source_note,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return {
        'id': r.id,
        'brand_id': r.brand_id,
        'kind': r.kind,
        'title': r.title,
        'pattern': r.pattern,
        'severity': r.severity,
        'explanation': r.explanation,
        'suggestion': r.suggestion,
        'source_note': r.source_note,
        'active': True,
    }


@app.post('/api/reviews', status_code=201)
def create_review(d: ReviewIn, db: Session = Depends(db_dep)):
    if not db.get(Brand, d.brand_id):
        raise HTTPException(404, 'Brand not found.')
    rules = db.scalars(select(Rule).where(Rule.brand_id == d.brand_id)).all()
    r = Review(brand_id=d.brand_id, title=d.title, content=d.content)
    db.add(r)
    db.flush()
    findings_data = run_engine(d.content, rules)
    for f in findings_data:
        db.add(Finding(review_id=r.id, **f))
    db.flush()
    r.status = calc_status(r.findings, '')
    db.commit()
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == r.id)
    )
    return review_json(r)


@app.get('/api/reviews')
def list_reviews(db: Session = Depends(db_dep)):
    rs = (
        db.scalars(
            select(Review)
            .options(joinedload(Review.findings))
            .order_by(Review.created_at.desc())
        )
        .unique()
        .all()
    )
    return [review_json(r) for r in rs]


@app.get('/api/reviews/{rid}')
def get_review(rid: int, db: Session = Depends(db_dep)):
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == rid)
    )
    if not r:
        raise HTTPException(404, 'Review not found.')
    return review_json(r)


@app.post('/api/reviews/{rid}/decisions')
def decide(rid: int, d: DecisionIn, db: Session = Depends(db_dep)):
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == rid)
    )
    if not r:
        raise HTTPException(404, 'Review not found.')
    if r.status == 'approved':
        raise HTTPException(
            400,
            'Cannot modify decisions on an already approved review. Reopen review'
            ' first.',
        )

    f = next((x for x in r.findings if x.id == d.finding_id), None)
    if not f:
        raise HTTPException(404, 'Finding not found in review.')

    target_decision = d.decision

    if target_decision == 'resolved':
        # Resolution requirements: If no suggested correction exists, require a reason
        resolution_note = d.reason.strip()
        if not resolution_note and not (
            f.suggested_correction and f.suggested_correction.strip()
        ):
            raise HTTPException(
                422,
                'Resolution requirement not met: Since no configured correction'
                ' exists for this rule, provide a decision reason explaining'
                ' how the violation was remediated.',
            )

    f.decision_status = target_decision
    f.decision_reason = d.reason.strip()

    if target_decision == 'resolved':
        f.evidence_status = 'verified_resolved'
    elif target_decision == 'rejected':
        f.evidence_status = 'dismissed'
    elif target_decision == 'accepted':
        f.evidence_status = 'confirmed_violation'

    db.add(
        Decision(
            review_id=r.id,
            finding_id=f.id,
            decision=target_decision,
            reason=d.reason.strip(),
        )
    )
    r.status = calc_status(r.findings, r.status)
    db.commit()
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == rid)
    )
    return review_json(r)


@app.post('/api/reviews/{rid}/approve')
def approve_review(rid: int, db: Session = Depends(db_dep)):
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == rid)
    )
    if not r:
        raise HTTPException(404, 'Review not found.')
    if r.status == 'approved':
        return review_json(r)
    if r.status != 'ready_for_approval':
        raise HTTPException(
            400,
            f"Cannot approve review: status is '{r.status}'. Review must be"
            " 'ready_for_approval' before it can be approved.",
        )
    r.status = 'approved'
    db.commit()
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == rid)
    )
    return review_json(r)


@app.post('/api/reviews/{rid}/reopen')
def reopen_review(rid: int, db: Session = Depends(db_dep)):
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == rid)
    )
    if not r:
        raise HTTPException(404, 'Review not found.')
    r.status = calc_status(r.findings, '')
    db.commit()
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == rid)
    )
    return review_json(r)


@app.get('/api/reviews/{rid}/report')
def get_review_report(rid: int, db: Session = Depends(db_dep)):
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == rid)
    )
    if not r:
        raise HTTPException(404, 'Review not found.')
    brand = db.get(Brand, r.brand_id)
    return generate_compliance_report(r, brand)


@app.get('/api/reviews/{rid}/passport')
def passport(rid: int, db: Session = Depends(db_dep)):
    r = db.scalar(
        select(Review)
        .options(joinedload(Review.findings))
        .where(Review.id == rid)
    )
    if not r:
        raise HTTPException(404, 'Review not found.')
    brand = db.get(Brand, r.brand_id)
    report = generate_compliance_report(r, brand)
    return {
        'report_type': 'VERITY Review Record',
        'disclaimer': (
            'This record documents checks against user-entered rules; it is not'
            ' independent proof of factual truth.'
        ),
        'review': review_json(r),
        'report': report,
    }
