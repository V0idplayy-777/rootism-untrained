"""The app-generation benchmark: realistic "build me an app" requests.

Each blueprint is a small, complete product specification - entities, typed
fields, relations, auth and one non-trivial feature - and it is the single
source of truth for three things:

* what :mod:`scaffold.blueprint` renders (schema, API, UI, tests, deploy files),
* what :mod:`dataset.app_corpus` trains on (the same prompts users type),
* what :mod:`evaluation.app_suite` measures (the fixed scoreboard).

Keeping one catalogue for train and eval is deliberate: the evaluation is
"can it build the apps people actually ask for", not a bespoke benchmark that
shares nothing with the product. Blueprints are held out from *training pairs*
by splitting on :func:`split`, not by writing different prompts.

Field types: ``text``, ``longtext``, ``int``, ``bool``, ``timestamp``,
``date``, ``enum``, ``ref``. A ``ref`` field names another entity and renders
as a foreign key with ``ON DELETE`` behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def pluralize(name: str) -> str:
    """English plural that is good enough for table/route names."""

    if not name:
        return name
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    return name + "s"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str = "text"
    label: str = ""
    required: bool = True
    choices: tuple[str, ...] = ()
    default: object = None
    help: str = ""
    ref: str = ""
    on_delete: str = "cascade"
    min_length: int = 0
    max_length: int = 0
    min_value: int | None = None
    max_value: int | None = None

    @property
    def display(self) -> str:
        return self.label or self.name.replace("_", " ").title()

    @property
    def column(self) -> str:
        return self.name

    @property
    def is_relation(self) -> bool:
        return self.type == "ref"


@dataclass(frozen=True)
class EntitySpec:
    name: str
    label: str
    fields: tuple[FieldSpec, ...]
    plural: str = ""
    display: str = "title"
    searchable: tuple[str, ...] = ()
    icon: str = "dot"

    @property
    def table(self) -> str:
        return self.plural or pluralize(self.name)

    @property
    def route(self) -> str:
        return self.table

    def field(self, name: str) -> FieldSpec | None:
        return next((item for item in self.fields if item.name == name), None)

    @property
    def relations(self) -> tuple[FieldSpec, ...]:
        return tuple(item for item in self.fields if item.is_relation)


@dataclass(frozen=True)
class AppBlueprint:
    id: str
    prompt: str
    title: str
    entities: tuple[EntitySpec, ...]
    persona: str = "Personal"
    feature: str = "none"
    feature_field: str = ""
    auth: bool = True
    accent: str = "#4f46e5"
    tags: tuple[str, ...] = ()
    notes: str = ""

    def entity(self, name: str) -> EntitySpec | None:
        return next((item for item in self.entities if item.name == name or item.plural == name), None)

    @property
    def primary(self) -> EntitySpec:
        """The entity the main list screen shows."""
        return self.entities[0]

    @property
    def tier(self) -> str:
        """Difficulty tier: plain CRUD, relations, or a non-trivial feature."""
        if self.feature not in {"none", ""}:
            return "feature"
        return "relations" if len(self.entities) > 1 else "crud"

    @property
    def flows(self) -> list[str]:
        """User-visible flows the tests and the suite must cover."""
        flows = ["signup", "login", "isolation"] if self.auth else ["open_access"]
        flows += ["list", "create", "edit", "delete", "search", "validation"]
        if len(self.entities) > 1:
            flows.append("relation")
        if self.feature == "board":
            flows.append("board")
        elif self.feature == "schedule":
            flows.append("conflict")
        elif self.feature == "team":
            flows.append("share")
        elif self.feature == "stats":
            flows.append("stats")
        return flows

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "title": self.title,
            "persona": self.persona,
            "feature": self.feature,
            "feature_field": self.feature_field,
            "auth": self.auth,
            "accent": self.accent,
            "tags": list(self.tags),
            "tier": self.tier,
            "entities": [
                {
                    "name": entity.name,
                    "label": entity.label,
                    "table": entity.table,
                    "display": entity.display,
                    "fields": [
                        {
                            "name": item.name,
                            "type": item.type,
                            "label": item.display,
                            "required": item.required,
                            "choices": list(item.choices),
                            "ref": item.ref,
                            "help": item.help,
                        }
                        for item in entity.fields
                    ],
                }
                for entity in self.entities
            ],
        }


def F(name: str, type: str = "text", label: str = "", **kwargs) -> FieldSpec:
    return FieldSpec(name=name, type=type, label=label, **kwargs)


def E(
    name: str,
    label: str,
    *fields: FieldSpec,
    display: str = "title",
    searchable: tuple[str, ...] = (),
    plural: str = "",
    icon: str = "dot",
) -> EntitySpec:
    return EntitySpec(
        name=name,
        label=label,
        fields=tuple(fields),
        display=display,
        searchable=searchable,
        plural=plural,
        icon=icon,
    )


def BP(
    id: str,
    prompt: str,
    title: str,
    entities: tuple[EntitySpec, ...],
    *,
    persona: str = "Personal",
    feature: str = "none",
    feature_field: str = "",
    auth: bool = True,
    accent: str = "#4f46e5",
    tags: tuple[str, ...] = (),
    notes: str = "",
) -> AppBlueprint:
    """Construct a blueprint without repeating keyword plumbing."""
    return AppBlueprint(
        id=id,
        prompt=prompt,
        title=title,
        entities=entities,
        persona=persona,
        feature=feature,
        feature_field=feature_field,
        auth=auth,
        accent=accent,
        tags=tags,
        notes=notes,
    )


DONE = ("todo", "doing", "done")
PRIORITY = ("low", "medium", "high")
STATUS = ("draft", "active", "archived")


def _task_entities() -> tuple[EntitySpec, ...]:
    return (
        E(
            "task",
            "Task",
            F("title", "text", "Title", max_length=140, help="What needs doing, in one line."),
            F("notes", "longtext", "Notes", required=False, max_length=4000),
            F("done", "bool", "Done", required=False, default=False),
            F("due_on", "date", "Due date", required=False),
            F("priority", "enum", "Priority", required=False, choices=PRIORITY, default="medium"),
            display="title",
            searchable=("title", "notes"),
            icon="check",
        ),
    )


BLUEPRINTS: tuple[AppBlueprint, ...] = (
    # ---------------------------------------------------------------- CRUD
    BP(
        "personal-tasks",
        "a simple to-do app for myself with due dates and priorities",
        "Personal Tasks",
        _task_entities(),
        auth=False,
        accent="#0f766e",
        tags=("crud", "productivity"),
    ),
    BP(
        "notes-app",
        "a notes app where I can pin important notes and search them",
        "Notes",
        (
            E(
                "note",
                "Note",
                F("title", "text", "Title", max_length=140),
                F("body", "longtext", "Body", max_length=8000),
                F("pinned", "bool", "Pinned", required=False, default=False),
                F("colour", "enum", "Colour", required=False, choices=("sand", "mint", "lilac", "sky"), default="sand"),
                display="title",
                searchable=("title", "body"),
                icon="note",
            ),
        ),
        persona="Personal",
        accent="#7c3aed",
        tags=("crud", "notes"),
    ),
    BP(
        "bookmark-vault",
        "a bookmark manager with tags and favourites that I can search",
        "Bookmarks",
        (
            E(
                "bookmark",
                "Bookmark",
                F("title", "text", "Title", max_length=160),
                F("url", "text", "URL", max_length=500, help="Full https:// URL."),
                F("tags", "text", "Tags", required=False, max_length=200, help="Comma separated."),
                F("favourite", "bool", "Favourite", required=False, default=False),
                display="title",
                searchable=("title", "url", "tags"),
                icon="bookmark",
            ),
        ),
        accent="#b45309",
        tags=("crud",),
    ),
    BP(
        "recipe-box",
        "a recipe box for my own cooking, with times and ingredients",
        "Recipe Box",
        (
            E(
                "recipe",
                "Recipe",
                F("title", "text", "Recipe", max_length=160),
                F("minutes", "int", "Minutes", required=False, min_value=1, max_value=600),
                F("ingredients", "longtext", "Ingredients", required=False, max_length=4000),
                F("favourite", "bool", "Favourite", required=False, default=False),
                display="title",
                searchable=("title", "ingredients"),
                icon="bowl",
            ),
        ),
        auth=False,
        accent="#be123c",
        tags=("crud", "food"),
    ),
    BP(
        "contact-book",
        "a contact book with companies and emails I can search",
        "Contacts",
        (
            E(
                "contact",
                "Contact",
                F("name", "text", "Name", max_length=120),
                F("email", "text", "Email", required=False, max_length=200),
                F("phone", "text", "Phone", required=False, max_length=40),
                F("company", "text", "Company", required=False, max_length=120),
                display="name",
                searchable=("name", "email", "company"),
                icon="person",
            ),
        ),
        accent="#0369a1",
        tags=("crud", "contacts"),
    ),
    BP(
        "expense-log",
        "an expense log with categories and a monthly total",
        "Expenses",
        (
            E(
                "expense",
                "Expense",
                F("title", "text", "What for", max_length=140),
                F("amount_cents", "int", "Amount (cents)", min_value=0, max_value=100_000_000),
                F("category", "enum", "Category", choices=("food", "travel", "home", "health", "fun"), default="food"),
                F("spent_on", "date", "Date", required=False),
                display="title",
                searchable=("title", "category"),
                icon="receipt",
            ),
        ),
        accent="#15803d",
        tags=("crud", "finance"),
    ),
    BP(
        "habit-tracker",
        "a habit tracker with weekly targets and an active toggle",
        "Habits",
        (
            E(
                "habit",
                "Habit",
                F("name", "text", "Habit", max_length=120),
                F("cadence", "enum", "Cadence", choices=("daily", "weekly"), default="daily"),
                F("target_per_week", "int", "Target per week", min_value=1, max_value=21),
                F("active", "bool", "Active", required=False, default=True),
                display="name",
                searchable=("name",),
                icon="spark",
            ),
        ),
        accent="#9333ea",
        tags=("crud", "habits"),
    ),
    BP(
        "reading-list",
        "a reading list that tracks what I am reading and my ratings",
        "Reading List",
        (
            E(
                "book",
                "Book",
                F("title", "text", "Title", max_length=200),
                F("author", "text", "Author", max_length=120),
                F("status", "enum", "Status", choices=("want", "reading", "finished"), default="want"),
                F("rating", "int", "Rating", required=False, min_value=1, max_value=5),
                display="title",
                searchable=("title", "author"),
                icon="book",
            ),
        ),
        accent="#1d4ed8",
        tags=("crud", "reading"),
    ),
    BP(
        "inventory-simple",
        "a small inventory app with quantities and reorder points",
        "Inventory",
        (
            E(
                "item",
                "Item",
                F("name", "text", "Item", max_length=140),
                F("sku", "text", "SKU", max_length=60),
                F("quantity", "int", "Quantity", min_value=0),
                F("reorder_at", "int", "Reorder at", required=False, min_value=0),
                display="name",
                searchable=("name", "sku"),
                icon="box",
            ),
        ),
        accent="#a16207",
        tags=("crud", "inventory"),
    ),
    BP(
        "job-applications",
        "a job application tracker with stages and follow-up dates",
        "Job Applications",
        (
            E(
                "application",
                "Application",
                F("company", "text", "Company", max_length=140),
                F("role", "text", "Role", max_length=140),
                F("status", "enum", "Status", choices=("applied", "interview", "offer", "rejected"), default="applied"),
                F("applied_on", "date", "Applied on", required=False),
                F("notes", "longtext", "Notes", required=False, max_length=4000),
                display="company",
                searchable=("company", "role", "notes"),
                icon="briefcase",
            ),
        ),
        accent="#0f766e",
        tags=("crud", "jobs"),
    ),
    BP(
        "workout-log",
        "a workout log with duration, intensity and the date",
        "Workout Log",
        (
            E(
                "workout",
                "Workout",
                F("name", "text", "Workout", max_length=140),
                F("minutes", "int", "Minutes", min_value=1, max_value=600),
                F("intensity", "enum", "Intensity", choices=("easy", "moderate", "hard"), default="moderate"),
                F("performed_on", "date", "Date", required=False),
                display="name",
                searchable=("name", "intensity"),
                icon="bolt",
            ),
        ),
        accent="#c2410c",
        tags=("crud", "fitness"),
    ),
    BP(
        "plant-care",
        "a plant care app that reminds me when to water each plant",
        "Plant Care",
        (
            E(
                "plant",
                "Plant",
                F("name", "text", "Plant", max_length=120),
                F("species", "text", "Species", required=False, max_length=120),
                F("water_every_days", "int", "Water every (days)", min_value=1, max_value=365),
                F("last_watered", "date", "Last watered", required=False),
                display="name",
                searchable=("name", "species"),
                icon="leaf",
            ),
        ),
        auth=False,
        accent="#166534",
        tags=("crud", "home"),
    ),
    # ----------------------------------------------------------- relations
    BP(
        "project-tasks",
        "a project tracker with projects and their tasks, with login",
        "Project Tasks",
        (
            E(
                "project",
                "Project",
                F("title", "text", "Project", max_length=140),
                F("status", "enum", "Status", choices=STATUS, default="active"),
                display="title",
                searchable=("title",),
                icon="folder",
            ),
            E(
                "task",
                "Task",
                F("project_id", "ref", "Project", ref="project", on_delete="cascade"),
                F("title", "text", "Task", max_length=160),
                F("done", "bool", "Done", required=False, default=False),
                F("due_on", "date", "Due date", required=False),
                display="title",
                searchable=("title",),
                icon="check",
            ),
        ),
        persona="Team",
        accent="#4f46e5",
        tags=("relations", "projects"),
    ),
    BP(
        "client-invoices",
        "an invoicing app with clients and their invoices",
        "Client Invoices",
        (
            E(
                "client",
                "Client",
                F("name", "text", "Client", max_length=140),
                F("email", "text", "Email", required=False, max_length=200),
                display="name",
                searchable=("name", "email"),
                icon="person",
            ),
            E(
                "invoice",
                "Invoice",
                F("client_id", "ref", "Client", ref="client", on_delete="cascade"),
                F("number", "text", "Invoice number", max_length=40),
                F("amount_cents", "int", "Amount (cents)", min_value=0),
                F("status", "enum", "Status", choices=("draft", "sent", "paid", "void"), default="draft"),
                F("due_on", "date", "Due date", required=False),
                display="number",
                searchable=("number", "status"),
                icon="receipt",
            ),
        ),
        persona="Freelancer",
        accent="#0e7490",
        tags=("relations", "finance"),
    ),
    BP(
        "course-lessons",
        "a course builder with courses and ordered lessons",
        "Course Lessons",
        (
            E(
                "course",
                "Course",
                F("title", "text", "Course", max_length=160),
                F("level", "enum", "Level", choices=("beginner", "intermediate", "advanced"), default="beginner"),
                display="title",
                searchable=("title",),
                icon="layers",
            ),
            E(
                "lesson",
                "Lesson",
                F("course_id", "ref", "Course", ref="course", on_delete="cascade"),
                F("title", "text", "Lesson", max_length=160),
                F("minutes", "int", "Minutes", required=False, min_value=1, max_value=600),
                F("position", "int", "Position", required=False, min_value=1),
                display="title",
                searchable=("title",),
                icon="play",
            ),
        ),
        accent="#6d28d9",
        tags=("relations", "education"),
    ),
    BP(
        "trip-itinerary",
        "a trip planner with trips and a day-by-day itinerary",
        "Trip Itinerary",
        (
            E(
                "trip",
                "Trip",
                F("name", "text", "Trip", max_length=140),
                F("starts_on", "date", "Starts", required=False),
                F("ends_on", "date", "Ends", required=False),
                display="name",
                searchable=("name",),
                icon="map",
            ),
            E(
                "stop",
                "Stop",
                F("trip_id", "ref", "Trip", ref="trip", on_delete="cascade"),
                F("city", "text", "City", max_length=120),
                F("day", "int", "Day number", required=False, min_value=1, max_value=60),
                F("notes", "longtext", "Notes", required=False, max_length=2000),
                display="city",
                searchable=("city", "notes"),
                icon="pin",
            ),
        ),
        accent="#b45309",
        tags=("relations", "travel"),
    ),
    BP(
        "warehouse-stock",
        "warehouse stock tracking across several warehouses",
        "Warehouse Stock",
        (
            E(
                "warehouse",
                "Warehouse",
                F("name", "text", "Warehouse", max_length=120),
                F("city", "text", "City", required=False, max_length=120),
                display="name",
                searchable=("name", "city"),
                icon="box",
            ),
            E(
                "stock_item",
                "Stock item",
                F("warehouse_id", "ref", "Warehouse", ref="warehouse", on_delete="cascade"),
                F("sku", "text", "SKU", max_length=60),
                F("quantity", "int", "Quantity", min_value=0),
                F("reorder_at", "int", "Reorder at", required=False, min_value=0),
                display="sku",
                searchable=("sku",),
                icon="box",
            ),
        ),
        persona="Operations",
        accent="#a16207",
        tags=("relations", "inventory"),
    ),
    BP(
        "pet-records",
        "a pet health log with pets and their vet visits",
        "Pet Records",
        (
            E(
                "pet",
                "Pet",
                F("name", "text", "Pet", max_length=120),
                F("species", "text", "Species", required=False, max_length=80),
                display="name",
                searchable=("name", "species"),
                icon="paw",
            ),
            E(
                "visit",
                "Visit",
                F("pet_id", "ref", "Pet", ref="pet", on_delete="cascade"),
                F("reason", "text", "Reason", max_length=160),
                F("visit_on", "date", "Visit date", required=False),
                F("cost_cents", "int", "Cost (cents)", required=False, min_value=0),
                display="reason",
                searchable=("reason",),
                icon="heart",
            ),
        ),
        auth=False,
        accent="#be185d",
        tags=("relations", "pets"),
    ),
    BP(
        "event-guests",
        "an event guest list with RSVP status per guest",
        "Event Guests",
        (
            E(
                "event",
                "Event",
                F("name", "text", "Event", max_length=160),
                F("starts_at", "timestamp", "Starts at", required=False),
                F("venue", "text", "Venue", required=False, max_length=160),
                display="name",
                searchable=("name", "venue"),
                icon="calendar",
            ),
            E(
                "guest",
                "Guest",
                F("event_id", "ref", "Event", ref="event", on_delete="cascade"),
                F("name", "text", "Guest", max_length=120),
                F("email", "text", "Email", required=False, max_length=200),
                F("rsvp", "enum", "RSVP", choices=("invited", "yes", "no", "maybe"), default="invited"),
                display="name",
                searchable=("name", "email"),
                icon="person",
            ),
        ),
        accent="#7c3aed",
        tags=("relations", "events"),
    ),
    BP(
        "sprint-issues",
        "a sprint board with sprints and issues assigned to each",
        "Sprint Issues",
        (
            E(
                "sprint",
                "Sprint",
                F("name", "text", "Sprint", max_length=120),
                F("goal", "text", "Goal", required=False, max_length=300),
                F("ends_on", "date", "Ends", required=False),
                display="name",
                searchable=("name", "goal"),
                icon="flag",
            ),
            E(
                "issue",
                "Issue",
                F("sprint_id", "ref", "Sprint", ref="sprint", on_delete="cascade"),
                F("title", "text", "Issue", max_length=180),
                F("priority", "enum", "Priority", choices=PRIORITY, default="medium"),
                F("status", "enum", "Status", choices=DONE, default="todo"),
                display="title",
                searchable=("title",),
                icon="bug",
            ),
        ),
        persona="Team",
        accent="#2563eb",
        tags=("relations", "engineering"),
    ),
    BP(
        "blog-comments",
        "a blog with posts and threaded comments",
        "Blog Comments",
        (
            E(
                "post",
                "Post",
                F("title", "text", "Title", max_length=200),
                F("body", "longtext", "Body", max_length=12000),
                F("published", "bool", "Published", required=False, default=False),
                display="title",
                searchable=("title", "body"),
                icon="note",
            ),
            E(
                "comment",
                "Comment",
                F("post_id", "ref", "Post", ref="post", on_delete="cascade"),
                F("author", "text", "Author", max_length=120),
                F("body", "longtext", "Comment", max_length=4000),
                display="author",
                searchable=("author", "body"),
                icon="chat",
            ),
        ),
        accent="#0f766e",
        tags=("relations", "content"),
    ),
    BP(
        "property-maintenance",
        "a property maintenance tracker with properties and jobs",
        "Property Maintenance",
        (
            E(
                "property",
                "Property",
                F("address", "text", "Address", max_length=240),
                F("tenant", "text", "Tenant", required=False, max_length=120),
                display="address",
                searchable=("address", "tenant"),
                icon="home",
            ),
            E(
                "job",
                "Job",
                F("property_id", "ref", "Property", ref="property", on_delete="cascade"),
                F("title", "text", "Job", max_length=180),
                F("status", "enum", "Status", choices=("open", "scheduled", "done"), default="open"),
                F("scheduled_on", "date", "Scheduled", required=False),
                display="title",
                searchable=("title",),
                icon="wrench",
            ),
        ),
        persona="Landlord",
        accent="#b45309",
        tags=("relations",),
    ),
    BP(
        "vehicle-service",
        "a vehicle service history with vehicles and service records",
        "Vehicle Service",
        (
            E(
                "vehicle",
                "Vehicle",
                F("make", "text", "Make", max_length=60),
                F("model", "text", "Model", max_length=60),
                F("plate", "text", "Plate", max_length=20),
                display="plate",
                searchable=("make", "model", "plate"),
                icon="car",
            ),
            E(
                "service",
                "Service",
                F("vehicle_id", "ref", "Vehicle", ref="vehicle", on_delete="cascade"),
                F("description", "text", "Description", max_length=200),
                F("cost_cents", "int", "Cost (cents)", required=False, min_value=0),
                F("serviced_on", "date", "Date", required=False),
                display="description",
                searchable=("description",),
                icon="wrench",
            ),
        ),
        auth=False,
        accent="#1d4ed8",
        tags=("relations",),
    ),
    BP(
        "class-roster",
        "a class roster with classes and enrolled students",
        "Class Roster",
        (
            E(
                "klass",
                "Class",
                F("name", "text", "Class", max_length=140),
                F("term", "text", "Term", required=False, max_length=60),
                display="name",
                searchable=("name", "term"),
                plural="classes",
                icon="layers",
            ),
            E(
                "enrollment",
                "Enrollment",
                F("klass_id", "ref", "Class", ref="klass", on_delete="cascade"),
                F("student_name", "text", "Student", max_length=140),
                F("email", "text", "Email", required=False, max_length=200),
                F("grade", "int", "Grade", required=False, min_value=0, max_value=100),
                display="student_name",
                searchable=("student_name", "email"),
                icon="person",
            ),
        ),
        persona="Teacher",
        accent="#0891b2",
        tags=("relations", "education"),
    ),
    BP(
        "sales-pipeline",
        "a sales pipeline with pipelines and deals in stages",
        "Sales Pipeline",
        (
            E(
                "pipeline",
                "Pipeline",
                F("name", "text", "Pipeline", max_length=120),
                display="name",
                searchable=("name",),
                icon="funnel",
            ),
            E(
                "deal",
                "Deal",
                F("pipeline_id", "ref", "Pipeline", ref="pipeline", on_delete="cascade"),
                F("company", "text", "Company", max_length=140),
                F("amount_cents", "int", "Amount (cents)", min_value=0),
                F("stage", "enum", "Stage", choices=("lead", "qualified", "proposal", "won", "lost"), default="lead"),
                display="company",
                searchable=("company", "stage"),
                icon="deal",
            ),
        ),
        persona="Sales",
        accent="#047857",
        tags=("relations", "sales"),
    ),
    BP(
        "budget-categories",
        "a monthly budget with categories and allocated amounts",
        "Budget",
        (
            E(
                "budget",
                "Budget",
                F("month", "text", "Month", max_length=20, help="For example 2026-11."),
                F("total_cents", "int", "Total (cents)", min_value=0),
                display="month",
                searchable=("month",),
                icon="wallet",
            ),
            E(
                "category",
                "Category",
                F("budget_id", "ref", "Budget", ref="budget", on_delete="cascade"),
                F("name", "text", "Category", max_length=80),
                F("allocated_cents", "int", "Allocated (cents)", min_value=0),
                display="name",
                searchable=("name",),
                icon="tag",
            ),
        ),
        accent="#0e7490",
        tags=("relations", "finance"),
    ),
    BP(
        "team-roster",
        "a team roster with teams and members and roles",
        "Team Roster",
        (
            E(
                "team",
                "Team",
                F("name", "text", "Team", max_length=120),
                F("description", "text", "Description", required=False, max_length=300),
                display="name",
                searchable=("name", "description"),
                icon="people",
            ),
            E(
                "member",
                "Member",
                F("team_id", "ref", "Team", ref="team", on_delete="cascade"),
                F("name", "text", "Name", max_length=120),
                F("email", "text", "Email", required=False, max_length=200),
                F("role", "enum", "Role", choices=("owner", "admin", "member"), default="member"),
                display="name",
                searchable=("name", "email"),
                icon="person",
            ),
        ),
        persona="Team",
        accent="#4338ca",
        tags=("relations", "people"),
    ),
    BP(
        "project-milestones",
        "a project plan with projects, milestones and tasks under each milestone",
        "Project Milestones",
        (
            E(
                "project",
                "Project",
                F("title", "text", "Project", max_length=140),
                display="title",
                searchable=("title",),
                icon="folder",
            ),
            E(
                "milestone",
                "Milestone",
                F("project_id", "ref", "Project", ref="project", on_delete="cascade"),
                F("title", "text", "Milestone", max_length=160),
                F("due_on", "date", "Due", required=False),
                display="title",
                searchable=("title",),
                icon="flag",
            ),
            E(
                "task",
                "Task",
                F("milestone_id", "ref", "Milestone", ref="milestone", on_delete="cascade"),
                F("title", "text", "Task", max_length=160),
                F("done", "bool", "Done", required=False, default=False),
                display="title",
                searchable=("title",),
                icon="check",
            ),
        ),
        persona="Team",
        accent="#1d4ed8",
        tags=("relations", "projects", "deep"),
    ),
    BP(
        "recipe-ingredients",
        "a recipe manager with recipes and structured ingredient lines",
        "Recipe Ingredients",
        (
            E(
                "recipe",
                "Recipe",
                F("title", "text", "Recipe", max_length=160),
                F("servings", "int", "Servings", min_value=1, max_value=50),
                display="title",
                searchable=("title",),
                icon="bowl",
            ),
            E(
                "ingredient",
                "Ingredient",
                F("recipe_id", "ref", "Recipe", ref="recipe", on_delete="cascade"),
                F("name", "text", "Ingredient", max_length=120),
                F("quantity", "int", "Quantity", min_value=1, max_value=10_000),
                F("unit", "text", "Unit", required=False, max_length=40),
                display="name",
                searchable=("name", "unit"),
                icon="leaf",
            ),
        ),
        accent="#be123c",
        tags=("relations", "food"),
    ),
    BP(
        "org-structure",
        "an org chart with organisations, teams and people",
        "Org Structure",
        (
            E(
                "org",
                "Organisation",
                F("name", "text", "Organisation", max_length=140),
                display="name",
                searchable=("name",),
                icon="building",
            ),
            E(
                "team",
                "Team",
                F("org_id", "ref", "Organisation", ref="org", on_delete="cascade"),
                F("name", "text", "Team", max_length=120),
                display="name",
                searchable=("name",),
                icon="people",
            ),
            E(
                "person",
                "Person",
                F("team_id", "ref", "Team", ref="team", on_delete="cascade"),
                F("name", "text", "Name", max_length=120),
                F("role", "text", "Role", required=False, max_length=120),
                display="name",
                searchable=("name", "role"),
                icon="person",
            ),
        ),
        persona="Team",
        accent="#475569",
        tags=("relations", "deep"),
    ),
    BP(
        "support-queues",
        "a support ticket system with queues and tickets",
        "Support Queues",
        (
            E(
                "queue",
                "Queue",
                F("name", "text", "Queue", max_length=120),
                display="name",
                searchable=("name",),
                icon="inbox",
            ),
            E(
                "ticket",
                "Ticket",
                F("queue_id", "ref", "Queue", ref="queue", on_delete="cascade"),
                F("subject", "text", "Subject", max_length=200),
                F("body", "longtext", "Body", required=False, max_length=6000),
                F("priority", "enum", "Priority", choices=PRIORITY, default="medium"),
                F("status", "enum", "Status", choices=("open", "pending", "closed"), default="open"),
                display="subject",
                searchable=("subject", "body"),
                icon="ticket",
            ),
        ),
        persona="Support",
        accent="#c2410c",
        tags=("relations", "support"),
    ),
    BP(
        "course-quizzes",
        "a quiz builder with courses, quizzes and questions",
        "Course Quizzes",
        (
            E(
                "course",
                "Course",
                F("title", "text", "Course", max_length=160),
                display="title",
                searchable=("title",),
                icon="layers",
            ),
            E(
                "quiz",
                "Quiz",
                F("course_id", "ref", "Course", ref="course", on_delete="cascade"),
                F("title", "text", "Quiz", max_length=160),
                display="title",
                searchable=("title",),
                icon="quiz",
            ),
            E(
                "question",
                "Question",
                F("quiz_id", "ref", "Quiz", ref="quiz", on_delete="cascade"),
                F("prompt", "text", "Question", max_length=300),
                F("answer", "text", "Answer", max_length=300),
                display="prompt",
                searchable=("prompt", "answer"),
                icon="help",
            ),
        ),
        accent="#6d28d9",
        tags=("relations", "education", "deep"),
    ),
    BP(
        "farm-fields",
        "a farm planner with farms, fields and crops",
        "Farm Fields",
        (
            E(
                "farm",
                "Farm",
                F("name", "text", "Farm", max_length=140),
                display="name",
                searchable=("name",),
                icon="leaf",
            ),
            E(
                "field",
                "Field",
                F("farm_id", "ref", "Farm", ref="farm", on_delete="cascade"),
                F("name", "text", "Field", max_length=120),
                F("area_hectares", "int", "Area (hectares)", required=False, min_value=1),
                display="name",
                searchable=("name",),
                icon="map",
            ),
            E(
                "crop",
                "Crop",
                F("field_id", "ref", "Field", ref="field", on_delete="cascade"),
                F("name", "text", "Crop", max_length=120),
                F("season", "enum", "Season", choices=("spring", "summer", "autumn", "winter"), default="spring"),
                display="name",
                searchable=("name",),
                icon="seed",
            ),
        ),
        auth=False,
        accent="#166534",
        tags=("relations", "deep"),
    ),
    BP(
        "restaurant-menu",
        "a restaurant menu editor with restaurants and dishes",
        "Restaurant Menu",
        (
            E(
                "restaurant",
                "Restaurant",
                F("name", "text", "Restaurant", max_length=160),
                F("city", "text", "City", required=False, max_length=120),
                display="name",
                searchable=("name", "city"),
                icon="building",
            ),
            E(
                "dish",
                "Dish",
                F("restaurant_id", "ref", "Restaurant", ref="restaurant", on_delete="cascade"),
                F("name", "text", "Dish", max_length=160),
                F("price_cents", "int", "Price (cents)", min_value=0),
                F("category", "enum", "Category", choices=("starter", "main", "dessert", "drink"), default="main"),
                display="name",
                searchable=("name", "category"),
                icon="bowl",
            ),
        ),
        accent="#b45309",
        tags=("relations", "food"),
    ),
    BP(
        "agency-projects",
        "an agency workspace with clients, projects and tasks",
        "Agency Projects",
        (
            E(
                "client",
                "Client",
                F("name", "text", "Client", max_length=140),
                F("contact_email", "text", "Contact email", required=False, max_length=200),
                display="name",
                searchable=("name", "contact_email"),
                icon="person",
            ),
            E(
                "project",
                "Project",
                F("client_id", "ref", "Client", ref="client", on_delete="cascade"),
                F("name", "text", "Project", max_length=160),
                F("status", "enum", "Status", choices=STATUS, default="active"),
                display="name",
                searchable=("name",),
                icon="folder",
            ),
            E(
                "task",
                "Task",
                F("project_id", "ref", "Project", ref="project", on_delete="cascade"),
                F("title", "text", "Task", max_length=160),
                F("done", "bool", "Done", required=False, default=False),
                display="title",
                searchable=("title",),
                icon="check",
            ),
        ),
        persona="Team",
        accent="#0e7490",
        tags=("relations", "deep"),
    ),
    # ------------------------------------------------------------- features
    BP(
        "task-board",
        "a kanban board for my tasks with todo, doing and done columns",
        "Task Board",
        (
            E(
                "task",
                "Task",
                F("title", "text", "Task", max_length=160),
                F("notes", "longtext", "Notes", required=False, max_length=3000),
                F("status", "enum", "Status", choices=DONE, default="todo"),
                F("priority", "enum", "Priority", choices=PRIORITY, default="medium"),
                display="title",
                searchable=("title", "notes"),
                icon="board",
            ),
        ),
        feature="board",
        feature_field="status",
        accent="#4f46e5",
        tags=("board", "productivity"),
    ),
    BP(
        "sprint-board",
        "a sprint board where each sprint has issues that move across columns",
        "Sprint Board",
        (
            E(
                "sprint",
                "Sprint",
                F("name", "text", "Sprint", max_length=120),
                F("starts_on", "date", "Starts", required=False),
                display="name",
                searchable=("name",),
                icon="flag",
            ),
            E(
                "issue",
                "Issue",
                F("sprint_id", "ref", "Sprint", ref="sprint", on_delete="cascade"),
                F("title", "text", "Issue", max_length=180),
                F("status", "enum", "Status", choices=DONE, default="todo"),
                F("priority", "enum", "Priority", choices=PRIORITY, default="medium"),
                display="title",
                searchable=("title",),
                icon="bug",
            ),
        ),
        feature="board",
        feature_field="status",
        persona="Team",
        accent="#2563eb",
        tags=("board", "engineering"),
    ),
    BP(
        "hiring-pipeline",
        "a hiring pipeline where candidates move through interview stages",
        "Hiring Pipeline",
        (
            E(
                "role",
                "Role",
                F("title", "text", "Role", max_length=140),
                F("department", "text", "Department", required=False, max_length=100),
                display="title",
                searchable=("title", "department"),
                icon="briefcase",
            ),
            E(
                "candidate",
                "Candidate",
                F("role_id", "ref", "Role", ref="role", on_delete="cascade"),
                F("name", "text", "Candidate", max_length=140),
                F("status", "enum", "Status", choices=("applied", "screen", "interview", "offer", "hired"), default="applied"),
                F("email", "text", "Email", required=False, max_length=200),
                display="name",
                searchable=("name", "email"),
                icon="person",
            ),
        ),
        feature="board",
        feature_field="status",
        persona="Hiring",
        accent="#7c3aed",
        tags=("board", "hiring"),
    ),
    BP(
        "support-board",
        "a support board where tickets move from open to resolved",
        "Support Board",
        (
            E(
                "ticket",
                "Ticket",
                F("subject", "text", "Subject", max_length=200),
                F("requester", "text", "Requester", max_length=140),
                F("status", "enum", "Status", choices=("new", "open", "waiting", "resolved"), default="new"),
                F("priority", "enum", "Priority", choices=PRIORITY, default="medium"),
                display="subject",
                searchable=("subject", "requester"),
                icon="ticket",
            ),
        ),
        feature="board",
        feature_field="status",
        persona="Support",
        accent="#c2410c",
        tags=("board", "support"),
    ),
    BP(
        "content-calendar",
        "a content calendar with posts moving from draft to published",
        "Content Calendar",
        (
            E(
                "post",
                "Post",
                F("title", "text", "Title", max_length=200),
                F("channel", "enum", "Channel", choices=("blog", "newsletter", "social"), default="blog"),
                F("status", "enum", "Status", choices=("idea", "drafting", "review", "published"), default="idea"),
                F("publish_on", "date", "Publish on", required=False),
                display="title",
                searchable=("title", "channel"),
                icon="calendar",
            ),
        ),
        feature="board",
        feature_field="status",
        accent="#0f766e",
        tags=("board", "content"),
    ),
    BP(
        "bug-triage",
        "a bug triage board with severity and status columns",
        "Bug Triage",
        (
            E(
                "bug",
                "Bug",
                F("title", "text", "Bug", max_length=200),
                F("severity", "enum", "Severity", choices=("low", "medium", "high", "critical"), default="medium"),
                F("status", "enum", "Status", choices=("triage", "confirmed", "fixing", "fixed"), default="triage"),
                F("reported_by", "text", "Reported by", required=False, max_length=120),
                display="title",
                searchable=("title", "reported_by"),
                icon="bug",
            ),
        ),
        feature="board",
        feature_field="status",
        persona="Engineering",
        accent="#b91c1c",
        tags=("board", "engineering"),
    ),
    BP(
        "order-fulfilment",
        "an order fulfilment board from paid to shipped",
        "Order Fulfilment",
        (
            E(
                "order",
                "Order",
                F("reference", "text", "Reference", max_length=60),
                F("customer", "text", "Customer", max_length=140),
                F("total_cents", "int", "Total (cents)", min_value=0),
                F("status", "enum", "Status", choices=("paid", "packed", "shipped", "delivered"), default="paid"),
                display="reference",
                searchable=("reference", "customer"),
                icon="box",
            ),
        ),
        feature="board",
        feature_field="status",
        persona="Operations",
        accent="#047857",
        tags=("board", "operations"),
    ),
    BP(
        "renovation-board",
        "a renovation project board with rooms moving through stages",
        "Renovation Board",
        (
            E(
                "room",
                "Room",
                F("name", "text", "Room", max_length=120),
                F("budget_cents", "int", "Budget (cents)", required=False, min_value=0),
                F("status", "enum", "Status", choices=("planned", "demo", "build", "done"), default="planned"),
                display="name",
                searchable=("name",),
                icon="home",
            ),
        ),
        feature="board",
        feature_field="status",
        auth=False,
        accent="#a16207",
        tags=("board", "home"),
    ),
    BP(
        "meeting-rooms",
        "book meeting rooms with conflict detection by room and time",
        "Meeting Rooms",
        (
            E(
                "room",
                "Room",
                F("name", "text", "Room", max_length=120),
                F("capacity", "int", "Capacity", min_value=1, max_value=200),
                display="name",
                searchable=("name",),
                icon="building",
            ),
            E(
                "booking",
                "Booking",
                F("room_id", "ref", "Room", ref="room", on_delete="cascade"),
                F("title", "text", "Meeting", max_length=160),
                F("start_at", "timestamp", "Starts", help="ISO timestamp."),
                F("end_at", "timestamp", "Ends", help="ISO timestamp."),
                display="title",
                searchable=("title",),
                icon="calendar",
            ),
        ),
        feature="schedule",
        persona="Team",
        accent="#0e7490",
        tags=("schedule", "workplace"),
    ),
    BP(
        "salon-appointments",
        "a salon appointment book that prevents double bookings",
        "Salon Appointments",
        (
            E(
                "stylist",
                "Stylist",
                F("name", "text", "Stylist", max_length=120),
                display="name",
                searchable=("name",),
                icon="person",
            ),
            E(
                "appointment",
                "Appointment",
                F("stylist_id", "ref", "Stylist", ref="stylist", on_delete="cascade"),
                F("client_name", "text", "Client", max_length=140),
                F("start_at", "timestamp", "Starts", help="ISO timestamp."),
                F("end_at", "timestamp", "Ends", help="ISO timestamp."),
                display="client_name",
                searchable=("client_name",),
                icon="calendar",
            ),
        ),
        feature="schedule",
        accent="#be185d",
        tags=("schedule", "services"),
    ),
    BP(
        "tutoring-sessions",
        "a tutoring scheduler that stops overlapping sessions",
        "Tutoring Sessions",
        (
            E(
                "tutor",
                "Tutor",
                F("name", "text", "Tutor", max_length=120),
                F("subject", "text", "Subject", required=False, max_length=80),
                display="name",
                searchable=("name", "subject"),
                icon="person",
            ),
            E(
                "session",
                "Session",
                F("tutor_id", "ref", "Tutor", ref="tutor", on_delete="cascade"),
                F("student_name", "text", "Student", max_length=140),
                F("start_at", "timestamp", "Starts", help="ISO timestamp."),
                F("end_at", "timestamp", "Ends", help="ISO timestamp."),
                display="student_name",
                searchable=("student_name",),
                icon="calendar",
            ),
        ),
        feature="schedule",
        accent="#6d28d9",
        tags=("schedule", "education"),
    ),
    BP(
        "equipment-rentals",
        "equipment rental bookings with no overlapping reservations",
        "Equipment Rentals",
        (
            E(
                "equipment",
                "Equipment",
                F("name", "text", "Equipment", max_length=140),
                F("daily_rate_cents", "int", "Daily rate (cents)", required=False, min_value=0),
                display="name",
                searchable=("name",),
                icon="tool",
            ),
            E(
                "rental",
                "Rental",
                F("equipment_id", "ref", "Equipment", ref="equipment", on_delete="cascade"),
                F("customer", "text", "Customer", max_length=140),
                F("start_at", "timestamp", "Starts", help="ISO timestamp."),
                F("end_at", "timestamp", "Ends", help="ISO timestamp."),
                display="customer",
                searchable=("customer",),
                icon="calendar",
            ),
        ),
        feature="schedule",
        persona="Operations",
        accent="#a16207",
        tags=("schedule", "operations"),
    ),
    BP(
        "clinic-slots",
        "clinic appointments where a practitioner cannot be double booked",
        "Clinic Slots",
        (
            E(
                "practitioner",
                "Practitioner",
                F("name", "text", "Practitioner", max_length=120),
                F("speciality", "text", "Speciality", required=False, max_length=80),
                display="name",
                searchable=("name", "speciality"),
                icon="person",
            ),
            E(
                "appointment",
                "Appointment",
                F("practitioner_id", "ref", "Practitioner", ref="practitioner", on_delete="cascade"),
                F("patient_name", "text", "Patient", max_length=140),
                F("start_at", "timestamp", "Starts", help="ISO timestamp."),
                F("end_at", "timestamp", "Ends", help="ISO timestamp."),
                display="patient_name",
                searchable=("patient_name",),
                icon="calendar",
            ),
        ),
        feature="schedule",
        accent="#0f766e",
        tags=("schedule", "health"),
    ),
    BP(
        "desk-bookings",
        "coworking desk bookings with conflict checks per desk",
        "Desk Bookings",
        (
            E(
                "desk",
                "Desk",
                F("name", "text", "Desk", max_length=80),
                F("zone", "text", "Zone", required=False, max_length=80),
                display="name",
                searchable=("name", "zone"),
                icon="building",
            ),
            E(
                "booking",
                "Booking",
                F("desk_id", "ref", "Desk", ref="desk", on_delete="cascade"),
                F("member_name", "text", "Member", max_length=140),
                F("start_at", "timestamp", "Starts", help="ISO timestamp."),
                F("end_at", "timestamp", "Ends", help="ISO timestamp."),
                display="member_name",
                searchable=("member_name",),
                icon="calendar",
            ),
        ),
        feature="schedule",
        persona="Team",
        accent="#334155",
        tags=("schedule", "workplace"),
    ),
    BP(
        "shared-tasks",
        "a team task list where I can share tasks with teammates",
        "Shared Tasks",
        (
            E(
                "task",
                "Task",
                F("title", "text", "Task", max_length=160),
                F("notes", "longtext", "Notes", required=False, max_length=3000),
                F("done", "bool", "Done", required=False, default=False),
                F("priority", "enum", "Priority", choices=PRIORITY, default="medium"),
                display="title",
                searchable=("title", "notes"),
                icon="check",
            ),
        ),
        feature="team",
        persona="Team",
        accent="#4f46e5",
        tags=("team", "collaboration"),
    ),
    BP(
        "shared-notes",
        "a shared notes space where notes can be shared with specific people",
        "Shared Notes",
        (
            E(
                "note",
                "Note",
                F("title", "text", "Title", max_length=160),
                F("body", "longtext", "Body", max_length=10000),
                display="title",
                searchable=("title", "body"),
                icon="note",
            ),
        ),
        feature="team",
        persona="Team",
        accent="#7c3aed",
        tags=("team", "notes"),
    ),
    BP(
        "family-groceries",
        "a family grocery list that several people can share and edit",
        "Family Groceries",
        (
            E(
                "item",
                "Item",
                F("name", "text", "Item", max_length=120),
                F("quantity", "int", "Quantity", min_value=1, max_value=999),
                F("bought", "bool", "Bought", required=False, default=False),
                display="name",
                searchable=("name",),
                icon="basket",
            ),
        ),
        feature="team",
        persona="Family",
        accent="#15803d",
        tags=("team", "home"),
    ),
    BP(
        "shared-goals",
        "a shared OKR tracker where goals can be shared with the team",
        "Shared Goals",
        (
            E(
                "goal",
                "Goal",
                F("title", "text", "Goal", max_length=180),
                F("owner_name", "text", "Owner", max_length=120),
                F("progress", "int", "Progress %", required=False, min_value=0, max_value=100),
                display="title",
                searchable=("title", "owner_name"),
                icon="flag",
            ),
        ),
        feature="team",
        persona="Team",
        accent="#0e7490",
        tags=("team", "planning"),
    ),
    BP(
        "shared-clients",
        "an agency client list shared between account managers",
        "Shared Clients",
        (
            E(
                "client",
                "Client",
                F("name", "text", "Client", max_length=160),
                F("contact_email", "text", "Contact email", required=False, max_length=200),
                F("status", "enum", "Status", choices=STATUS, default="active"),
                display="name",
                searchable=("name", "contact_email"),
                icon="person",
            ),
        ),
        feature="team",
        persona="Agency",
        accent="#b45309",
        tags=("team", "crm"),
    ),
    BP(
        "shared-inbox",
        "a shared support inbox where teammates can pick up tickets",
        "Shared Inbox",
        (
            E(
                "ticket",
                "Ticket",
                F("subject", "text", "Subject", max_length=200),
                F("requester", "text", "Requester", max_length=140),
                F("status", "enum", "Status", choices=("open", "pending", "closed"), default="open"),
                display="subject",
                searchable=("subject", "requester"),
                icon="inbox",
            ),
        ),
        feature="team",
        persona="Support",
        accent="#c2410c",
        tags=("team", "support"),
    ),
    BP(
        "book-club",
        "a book club shelf where members share books and ratings",
        "Book Club",
        (
            E(
                "book",
                "Book",
                F("title", "text", "Title", max_length=200),
                F("author", "text", "Author", max_length=140),
                F("rating", "int", "Rating", required=False, min_value=1, max_value=5),
                display="title",
                searchable=("title", "author"),
                icon="book",
            ),
        ),
        feature="team",
        persona="Community",
        accent="#1d4ed8",
        tags=("team", "reading"),
    ),
    BP(
        "volunteer-roster",
        "a volunteer roster that coordinators can share with each other",
        "Volunteer Roster",
        (
            E(
                "shift",
                "Shift",
                F("name", "text", "Shift", max_length=140),
                F("volunteer_name", "text", "Volunteer", max_length=140),
                F("starts_on", "date", "Date", required=False),
                F("hours", "int", "Hours", required=False, min_value=1, max_value=24),
                display="name",
                searchable=("name", "volunteer_name"),
                icon="clock",
            ),
        ),
        feature="team",
        persona="Community",
        accent="#166534",
        tags=("team", "community"),
    ),
    BP(
        "expense-dashboard",
        "an expense dashboard with totals by category and a monthly trend",
        "Expense Dashboard",
        (
            E(
                "expense",
                "Expense",
                F("title", "text", "Expense", max_length=140),
                F("amount_cents", "int", "Amount (cents)", min_value=0),
                F("category", "enum", "Category", choices=("food", "travel", "home", "health", "fun"), default="food"),
                F("spent_on", "date", "Date", required=False),
                display="title",
                searchable=("title", "category"),
                icon="receipt",
            ),
        ),
        feature="stats",
        feature_field="category",
        accent="#15803d",
        tags=("stats", "finance"),
    ),
    BP(
        "habit-dashboard",
        "a habit dashboard with completion counts and a weekly trend",
        "Habit Dashboard",
        (
            E(
                "habit_log",
                "Habit log",
                F("habit", "text", "Habit", max_length=120),
                F("completed", "bool", "Completed", required=False, default=True),
                F("mood", "enum", "Mood", choices=("low", "ok", "great"), default="ok"),
                F("logged_on", "date", "Date", required=False),
                display="habit",
                searchable=("habit",),
                icon="spark",
            ),
        ),
        feature="stats",
        feature_field="mood",
        accent="#9333ea",
        tags=("stats", "habits"),
    ),
    BP(
        "sales-dashboard",
        "a sales dashboard with revenue by stage and a trend line",
        "Sales Dashboard",
        (
            E(
                "deal",
                "Deal",
                F("company", "text", "Company", max_length=160),
                F("amount_cents", "int", "Amount (cents)", min_value=0),
                F("stage", "enum", "Stage", choices=("lead", "qualified", "proposal", "won", "lost"), default="lead"),
                F("closed_on", "date", "Date", required=False),
                display="company",
                searchable=("company", "stage"),
                icon="deal",
            ),
        ),
        feature="stats",
        feature_field="stage",
        persona="Sales",
        accent="#047857",
        tags=("stats", "sales"),
    ),
    BP(
        "survey-results",
        "a survey response viewer with counts per answer",
        "Survey Results",
        (
            E(
                "response",
                "Response",
                F("respondent", "text", "Respondent", max_length=140),
                F("answer", "enum", "Answer", choices=("yes", "no", "maybe"), default="yes"),
                F("score", "int", "Score", required=False, min_value=1, max_value=10),
                F("submitted_on", "date", "Date", required=False),
                display="respondent",
                searchable=("respondent",),
                icon="chart",
            ),
        ),
        feature="stats",
        feature_field="answer",
        accent="#0e7490",
        tags=("stats",),
    ),
    BP(
        "gym-tracker",
        "a gym tracker with sessions per week and intensity breakdown",
        "Gym Tracker",
        (
            E(
                "session",
                "Session",
                F("name", "text", "Session", max_length=140),
                F("intensity", "enum", "Intensity", choices=("easy", "moderate", "hard"), default="moderate"),
                F("minutes", "int", "Minutes", min_value=1, max_value=600),
                F("performed_on", "date", "Date", required=False),
                display="name",
                searchable=("name",),
                icon="bolt",
            ),
        ),
        feature="stats",
        feature_field="intensity",
        auth=False,
        accent="#c2410c",
        tags=("stats", "fitness"),
    ),
    BP(
        "reading-stats",
        "reading stats with books finished per genre",
        "Reading Stats",
        (
            E(
                "book",
                "Book",
                F("title", "text", "Title", max_length=200),
                F("genre", "enum", "Genre", choices=("fiction", "history", "science", "biography"), default="fiction"),
                F("pages", "int", "Pages", required=False, min_value=1),
                F("finished_on", "date", "Finished", required=False),
                display="title",
                searchable=("title",),
                icon="book",
            ),
        ),
        feature="stats",
        feature_field="genre",
        accent="#1d4ed8",
        tags=("stats", "reading"),
    ),
    BP(
        "job-search-stats",
        "job search stats showing applications per stage",
        "Job Search Stats",
        (
            E(
                "application",
                "Application",
                F("company", "text", "Company", max_length=160),
                F("stage", "enum", "Stage", choices=("applied", "interview", "offer", "rejected"), default="applied"),
                F("applied_on", "date", "Date", required=False),
                display="company",
                searchable=("company",),
                icon="briefcase",
            ),
        ),
        feature="stats",
        feature_field="stage",
        accent="#0f766e",
        tags=("stats", "jobs"),
    ),
    BP(
        "energy-usage",
        "an energy usage dashboard with usage by room and a monthly trend",
        "Energy Usage",
        (
            E(
                "reading",
                "Reading",
                F("room", "enum", "Room", choices=("kitchen", "living", "office", "garage"), default="kitchen"),
                F("kwh", "int", "kWh", min_value=0, max_value=100_000),
                F("read_on", "date", "Date", required=False),
                display="room",
                searchable=("room",),
                icon="bolt",
            ),
        ),
        feature="stats",
        feature_field="room",
        auth=False,
        accent="#a16207",
        tags=("stats", "home"),
    ),
)


BY_ID: dict[str, AppBlueprint] = {item.id: item for item in BLUEPRINTS}


def get_blueprint(blueprint_id: str) -> AppBlueprint:
    try:
        return BY_ID[blueprint_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown blueprint {blueprint_id!r}; choose from: {', '.join(sorted(BY_ID))}"
        ) from exc


def all_blueprints() -> list[AppBlueprint]:
    return list(BLUEPRINTS)


def split(name: str) -> tuple[list[AppBlueprint], list[AppBlueprint]]:
    """Deterministic train/eval split over the same catalogue.

    Eval blueprints are *held out of training pairs* by id (stable hash), so the
    scoreboard can never be improved by memorising the evaluation prompts.
    """

    import hashlib

    train: list[AppBlueprint] = []
    held_out: list[AppBlueprint] = []
    for item in BLUEPRINTS:
        digest = int(hashlib.sha256(item.id.encode()).hexdigest()[:8], 16)
        (held_out if digest % 5 == 0 else train).append(item)
    return train, held_out


def stats() -> dict:
    """Catalogue counts used by docs, the mixture and the manifest."""
    by_tier: dict[str, int] = {}
    by_feature: dict[str, int] = {}
    for item in BLUEPRINTS:
        by_tier[item.tier] = by_tier.get(item.tier, 0) + 1
        by_feature[item.feature] = by_feature.get(item.feature, 0) + 1
    return {
        "blueprints": len(BLUEPRINTS),
        "by_tier": by_tier,
        "by_feature": by_feature,
        "with_auth": sum(1 for item in BLUEPRINTS if item.auth),
        "entities_total": sum(len(item.entities) for item in BLUEPRINTS),
        "fields_total": sum(len(entity.fields) for item in BLUEPRINTS for entity in item.entities),
    }


MATCH_HINTS: dict[str, tuple[str, ...]] = {
    "personal-tasks": ("todo", "to-do", "task list", "to do"),
    "notes-app": ("note", "notes"),
    "bookmark-vault": ("bookmark", "link saver"),
    "recipe-box": ("recipe box", "cookbook"),
    "contact-book": ("contact", "address book"),
    "expense-log": ("expense log", "expenses"),
    "habit-tracker": ("habit",),
    "reading-list": ("reading list", "books"),
    "inventory-simple": ("inventory", "stock"),
    "job-applications": ("job application", "job search"),
    "workout-log": ("workout", "exercise"),
    "plant-care": ("plant",),
    "project-tasks": ("project", "tasks"),
    "client-invoices": ("invoice", "billing"),
    "course-lessons": ("course", "lesson"),
    "trip-itinerary": ("trip", "itinerary", "travel"),
    "warehouse-stock": ("warehouse",),
    "pet-records": ("pet", "vet"),
    "event-guests": ("guest list", "rsvp", "event"),
    "sprint-issues": ("sprint", "issue"),
    "blog-comments": ("blog", "post", "comment"),
    "property-maintenance": ("property", "maintenance", "landlord"),
    "vehicle-service": ("vehicle", "car service"),
    "class-roster": ("roster", "class", "student"),
    "sales-pipeline": ("sales pipeline", "deal"),
    "budget-categories": ("budget",),
    "team-roster": ("roster", "member"),
    "project-milestones": ("milestone",),
    "recipe-ingredients": ("ingredient",),
    "org-structure": ("org chart", "organisation", "organization"),
    "support-queues": ("queue", "ticket"),
    "course-quizzes": ("quiz", "question"),
    "farm-fields": ("farm", "field", "crop"),
    "restaurant-menu": ("menu", "restaurant", "dish"),
    "agency-projects": ("agency", "client", "project"),
    "task-board": ("kanban", "board", "column"),
    "sprint-board": ("sprint board",),
    "hiring-pipeline": ("hiring", "candidate", "recruit"),
    "support-board": ("support board", "ticket board"),
    "content-calendar": ("content calendar", "editorial"),
    "bug-triage": ("bug", "triage"),
    "order-fulfilment": ("order", "fulfilment", "fulfillment", "shipping"),
    "renovation-board": ("renovation", "remodel"),
    "meeting-rooms": ("meeting room", "room booking"),
    "salon-appointments": ("salon", "appointment"),
    "tutoring-sessions": ("tutor",),
    "equipment-rentals": ("rental", "equipment"),
    "clinic-slots": ("clinic", "patient", "doctor"),
    "desk-bookings": ("desk", "coworking"),
    "shared-tasks": ("share", "team task", "teammate"),
    "shared-notes": ("shared note",),
    "family-groceries": ("grocery", "shopping list", "family"),
    "shared-goals": ("okr", "goal"),
    "shared-clients": ("shared client", "account manager"),
    "shared-inbox": ("shared inbox",),
    "book-club": ("book club",),
    "volunteer-roster": ("volunteer", "shift"),
    "expense-dashboard": ("expense dashboard", "spending"),
    "habit-dashboard": ("habit dashboard", "streak"),
    "sales-dashboard": ("sales dashboard", "revenue"),
    "survey-results": ("survey", "responses"),
    "gym-tracker": ("gym", "fitness"),
    "reading-stats": ("reading stats", "genre"),
    "job-search-stats": ("job stats", "application stats"),
    "energy-usage": ("energy", "kwh"),
}


def match_blueprint(prompt: str, minimum: int = 2) -> AppBlueprint | None:
    """Pick the closest blueprint for a prompt with a transparent score.

    The matcher is deliberately simple and deterministic (keyword coverage plus
    a feature-word bonus). When nothing scores well the caller should fall back
    to the generic starters rather than pretending a data app fits.
    """

    text = " ".join((prompt or "").lower().replace("-", " ").split())
    if not text:
        return None
    best: tuple[int, AppBlueprint] | None = None
    for item in BLUEPRINTS:
        score = 0
        hints = MATCH_HINTS.get(item.id, ())
        for hint in hints:
            if hint in text:
                score += 3 if " " in hint else 2
        for word in item.prompt.lower().replace("-", " ").split():
            if len(word) > 4 and word in text:
                score += 1
        if item.feature != "none" and item.feature in text:
            score += 2
        if item.feature == "team" and any(word in text for word in ("invite", "share", "team", "together")):
            score += 2
        if item.feature == "board" and any(word in text for word in ("kanban", "column", "stage", "board")):
            score += 2
        if item.feature == "schedule" and any(word in text for word in ("book", "booking", "appointment", "calendar", "conflict")):
            score += 2
        if item.feature == "stats" and any(word in text for word in ("dashboard", "stats", "report", "chart", "trend")):
            score += 2
        if item.auth and any(word in text for word in ("login", "sign up", "signup", "account", "user", "teammate")):
            score += 1
        if best is None or score > best[0] or (score == best[0] and item.id < best[1].id):
            best = (score, item)
    if best is None or best[0] < minimum:
        return None
    return best[1]
