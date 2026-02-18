from sqlalchemy import String, Boolean, Enum
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database_session import Base
from src.api.role import UserRole


class User(Base):
    """
    Table definition for table called "users".
    """
    # Define table name
    __tablename__ = "users"

    # Define column named id as primary key
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(
        String(320),        # Sets max string length to 320 chars, to be valid
        unique=True,        # Only able to add emails that are unique -> not already existing in table
        index=True,         # Mappes id to each unique mail address -> w.x@y.z = number -> faster lookup  
        nullable=False,     # If adding a email address to the table, the val is not allowed to be None
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),        # Sets max string lenght to 255 chars
        nullable=False,     # The given string must not be None
    )
    # Marks if the user has access to the system or not. False -> No Access 
    is_active: Mapped[bool] = mapped_column(
        Boolean, 
        default=True,       # Default value is true, if no value given for a user 
        nullable=False      # If adding commadn sets is_active = None, then this will be declared as not valid
                            # and due to "default=True" the value will become True, instead of None
    )
    role: Mapped[UserRole] = mapped_column(
        # Stores the user role as string in the DB, but allows to work with it as Enum in the code 
        # -> more convenient and less error prone
        # SO in SQL world, we store the names (USER, DEVELOPER, ADMIN)
        # In Python world, it behaves like an enum, str class, which values we got by UserRole.User.value
        # and its names by UserRole.User.name
        Enum(UserRole),
        default=UserRole.USER,
        nullable=False,
    )

